#!/usr/bin/env python3
"""
Chat server for the king model. Runs on GPU pod, port 8100.
Features: SSE streaming, thinking/answer split, concurrent requests, no token cap.
~8GB VRAM for a 4B model. HF transformers backend (~37 tok/s on B200).
"""
import json
import sys
import time
import re
import torch
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "aceini/q-dist"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8100

print(f"[chat] Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
vram_gb = round(torch.cuda.memory_allocated() / 1e9, 1)
print(f"[chat] Model loaded. VRAM: {vram_gb}GB", flush=True)

_gen_lock = threading.Lock()


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 2048)  # No hard cap
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
            parts.append("assistant:")
            text = "\n".join(parts)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            repetition_penalty=1.1,
        )

        if stream:
            self._stream_response(gen_kwargs, input_len)
        else:
            self._sync_response(gen_kwargs, input_len)

    def _sync_response(self, gen_kwargs, input_len):
        t0 = time.time()
        with _gen_lock:
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
        elapsed = time.time() - t0
        new_tokens = output[0][input_len:]
        n_tokens = len(new_tokens)
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        tps = n_tokens / elapsed if elapsed > 0 else 0

        thinking, answer = _split_thinking(raw)

        result = {
            "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "model": MODEL_NAME,
            "usage": {"completion_tokens": n_tokens, "tokens_per_second": round(tps, 1), "generation_time_s": round(elapsed, 2)},
        }
        if thinking:
            result["thinking"] = thinking

        self._send_json(200, result)

    def _stream_response(self, gen_kwargs, input_len):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
        gen_kwargs["streamer"] = streamer

        t0 = time.time()
        n_tokens = [0]
        full_text = []
        phase = ["thinking"]  # start assuming thinking
        think_done = [False]

        def generate():
            with _gen_lock:
                with torch.no_grad():
                    model.generate(**gen_kwargs)

        thread = threading.Thread(target=generate)
        thread.start()

        try:
            for chunk in streamer:
                full_text.append(chunk)
                joined = "".join(full_text)
                n_tokens[0] += max(1, len(tokenizer.encode(chunk, add_special_tokens=False)))
                elapsed = time.time() - t0
                tps = n_tokens[0] / elapsed if elapsed > 0 else 0

                # Detect phase transitions
                current_phase = phase[0]
                if not think_done[0]:
                    if "</think>" in joined:
                        think_done[0] = True
                        phase[0] = "answer"
                        # Split: send the answer portion only
                        after_think = joined.split("</think>", 1)[1]
                        # Send transition event
                        self._sse({"choices": [{"delta": {"phase": "answer"}, "finish_reason": None}], "usage": {"tokens_per_second": round(tps, 1)}})
                        if after_think.strip():
                            self._sse({"choices": [{"delta": {"content": after_think.strip(), "phase": "answer"}, "finish_reason": None}], "usage": {"tokens_per_second": round(tps, 1)}})
                        continue
                    # Check if model didn't use think tags at all (first 50 chars)
                    if len(joined) > 50 and "<think>" not in joined[:50]:
                        think_done[0] = True
                        phase[0] = "answer"

                # Clean chunk: strip <think> tag from output
                clean = chunk.replace("<think>", "").replace("</think>", "")
                if not clean:
                    continue

                self._sse({
                    "choices": [{"delta": {"content": clean, "phase": phase[0]}, "finish_reason": None}],
                    "usage": {"tokens_per_second": round(tps, 1)},
                })
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            thread.join()

        elapsed = time.time() - t0
        tps = n_tokens[0] / elapsed if elapsed > 0 else 0
        try:
            self._sse({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": n_tokens[0], "tokens_per_second": round(tps, 1), "generation_time_s": round(elapsed, 2)},
            })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self, data):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok", "model": MODEL_NAME,
                "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
            })
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def _split_thinking(text):
    """Split <think>...</think> from answer."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return thinking, answer if answer else "(stopped during thinking)"
    if text.lstrip().startswith("<think>"):
        return text.lstrip()[7:].strip(), "(thinking cut short)"
    # Heuristic fallback for "Thinking Process:" style
    for header in ["Thinking Process:", "Thought:", "Reasoning:"]:
        if text.startswith(header):
            parts = re.split(r'\n\n(?=[A-Z*])', text, maxsplit=1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    return None, text


class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address), daemon=True)
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


if __name__ == "__main__":
    print(f"[chat] Serving on port {PORT} (threaded, streaming)", flush=True)
    server = ThreadedHTTPServer(("0.0.0.0", PORT), ChatHandler)
    server.serve_forever()
