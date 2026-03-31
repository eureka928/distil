#!/usr/bin/env python3
"""
KL distillation training — online knowledge distillation on 1x A100 80GB.

Teacher loaded in 4-bit (bitsandbytes NF4, ~18GB) for inference only.
Student loaded in bf16 (~8GB) with gradient checkpointing.
8-bit AdamW optimizer to fit within VRAM budget.

Loss = KL(teacher || student) on full 248K vocabulary, directly optimizing
what the validator measures.

Usage:
    python scripts/train_distill.py \\
        --student-repo Qwen/Qwen3.5-4B \\
        --output-dir ./checkpoints/run1 \\
        --num-steps 10000 \\
        --batch-size 2 --grad-accum 8 \\
        --lr 1e-5

    # Resume from checkpoint:
    python scripts/train_distill.py \\
        --student-repo Qwen/Qwen3.5-4B \\
        --output-dir ./checkpoints/run1 \\
        --resume-from ./checkpoints/run1/step-5000 \\
        --num-steps 10000

Requirements:
    pip install -e ".[train]"
"""
import sys
import gc
import json
import math
import random
import time
import logging
from pathlib import Path

import click
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.model_checker import TEACHER_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_distill")


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_vram(label: str = ""):
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        prefix = f"VRAM [{label}]" if label else "VRAM"
        logger.info(f"{prefix}: {used:.1f}GB allocated, {reserved:.1f}GB reserved / {total:.1f}GB total")


def compute_kl_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) averaged over all positions. Full 248K vocabulary."""
    t_log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    s_log_p = F.log_softmax(student_logits.float(), dim=-1)
    t_p = t_log_p.exp()
    kl_per_pos = (t_p * (t_log_p - s_log_p)).sum(dim=-1)  # [batch, seq_len]
    return kl_per_pos.mean()


def compute_continuation_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """KL(teacher || student) on continuation positions only, matching validator slicing.

    Uses the same slice as the validator: logits[:, prompt_len-1:-1, :]
    """
    start = max(prompt_len - 1, 0)
    seq_len = teacher_logits.shape[-2]

    if start >= seq_len - 1:
        logger.warning(f"No continuation positions (prompt_len={prompt_len}, seq_len={seq_len}), skipping")
        return torch.tensor(0.0, device=teacher_logits.device, requires_grad=True)

    # Slice to continuation positions before softmax (saves compute on prompt positions)
    t_cont = teacher_logits[:, start:-1, :]
    s_cont = student_logits[:, start:-1, :]

    t_log_p = F.log_softmax(t_cont.float(), dim=-1)
    s_log_p = F.log_softmax(s_cont.float(), dim=-1)
    t_p = t_log_p.exp()
    kl_per_pos = (t_p * (t_log_p - s_log_p)).sum(dim=-1)
    return kl_per_pos.mean()


def create_data_iterator(tokenizer, seq_len: int, seed: int = 42):
    """Stream FineWeb data, tokenized and chunked to seq_len."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    buffer_ids = []
    for example in ds:
        text = example.get("text", "")
        if len(text) < 200:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        buffer_ids.extend(ids)

        # Yield full chunks
        while len(buffer_ids) >= seq_len:
            chunk = buffer_ids[:seq_len]
            buffer_ids = buffer_ids[seq_len:]
            yield torch.tensor([chunk], dtype=torch.long)


def create_continuation_data_iterator(cache_dir: str, seed: int = 42):
    """Load pre-generated teacher continuations from cache. Yields (full_ids, prompt_len)."""
    cache_path = Path(cache_dir)
    sample_files = sorted(cache_path.glob("*.pt"))
    if not sample_files:
        raise FileNotFoundError(f"No .pt files found in {cache_dir}. Run generate_train_cache.py first.")

    rng = random.Random(seed)
    while True:
        shuffled = list(sample_files)
        rng.shuffle(shuffled)
        for f in shuffled:
            cached = torch.load(f, weights_only=True)
            yield cached["full_ids"], cached["prompt_len"]


def save_checkpoint(student, optimizer, step, output_dir):
    """Save student model + optimizer + training state."""
    ckpt_dir = Path(output_dir) / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    student.save_pretrained(ckpt_dir, safe_serialization=True)
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }, ckpt_dir / "training_state.pt")

    logger.info(f"Checkpoint saved: {ckpt_dir}")


@click.command()
@click.option("--student-repo", required=True,
              help="HuggingFace repo or local path for base student model")
@click.option("--student-revision", default="main")
@click.option("--teacher-model", default=TEACHER_MODEL)
@click.option("--output-dir", required=True, type=click.Path(),
              help="Directory for checkpoints")
@click.option("--resume-from", default=None, type=click.Path(),
              help="Resume from a checkpoint directory")
@click.option("--num-steps", type=int, default=10000,
              help="Total training steps")
@click.option("--batch-size", type=int, default=2,
              help="Per-device batch size (sequences per forward pass)")
@click.option("--grad-accum", type=int, default=8,
              help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
@click.option("--lr", type=float, default=1e-5, help="Learning rate")
@click.option("--warmup-steps", type=int, default=100, help="Linear warmup steps")
@click.option("--seq-len", type=int, default=1024, help="Sequence length (Phase 1 only)")
@click.option("--mode", type=click.Choice(["full", "continuation"]), default="full",
              help="full=Phase 1 (all positions on FineWeb), continuation=Phase 2 (continuation-only KL)")
@click.option("--cache-dir", default=None, type=click.Path(),
              help="Teacher continuation cache dir (required for --mode continuation)")
@click.option("--teacher-precision", type=click.Choice(["bf16", "4bit"]), default="4bit",
              help="Teacher precision: bf16 (B200/H100, best quality) or 4bit (A100, saves VRAM)")
@click.option("--save-every", type=int, default=500, help="Save checkpoint every N steps")
@click.option("--log-every", type=int, default=10, help="Log loss every N steps")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--device", default="cuda")
def main(
    student_repo, student_revision, teacher_model, output_dir, resume_from,
    num_steps, batch_size, grad_accum, lr, warmup_steps, seq_len, mode, cache_dir,
    teacher_precision, save_every, log_every, seed, device,
):
    """Train a student model via KL distillation against the teacher."""

    if teacher_precision == "4bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            logger.error("bitsandbytes not installed. Run: pip install -e '.[train]'")
            sys.exit(1)
    else:
        try:
            import bitsandbytes as bnb
        except ImportError:
            bnb = None  # Not needed for bf16 teacher

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install -e '.[train]'")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    effective_batch = batch_size * grad_accum
    logger.info(f"Training config:")
    logger.info(f"  Student: {student_repo}")
    logger.info(f"  Teacher: {teacher_model} ({teacher_precision})")
    logger.info(f"  Steps: {num_steps}, LR: {lr}, Warmup: {warmup_steps}")
    logger.info(f"  Batch: {batch_size} x {grad_accum} grad_accum = {effective_batch} effective")
    logger.info(f"  Seq len: {seq_len}, Mode: {mode}")
    logger.info(f"  Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load teacher ──────────────────────────────────────────────────
    logger.info(f"Loading teacher model ({teacher_precision})...")
    if teacher_precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log_vram("after teacher load")

    # ── Load student (bf16, trainable) ────────────────────────────────
    logger.info(f"Loading student model (bf16): {student_repo}")
    student = AutoModelForCausalLM.from_pretrained(
        resume_from or student_repo,
        revision=None if resume_from else student_revision,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    student.gradient_checkpointing_enable()
    student.train()
    log_vram("after student load")

    # ── Optimizer ─────────────────────────────────────────────────────
    if bnb is not None:
        optimizer = bnb.optim.AdamW8bit(
            student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
        )

    # Resume optimizer state if available
    start_step = 0
    if resume_from:
        state_path = Path(resume_from) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, weights_only=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_step = state["step"]
            logger.info(f"Resumed from step {start_step}")

    log_vram("after optimizer init")

    # ── LR schedule (linear warmup + cosine decay) ────────────────────
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    # ── Data iterator ─────────────────────────────────────────────────
    if mode == "continuation":
        if not cache_dir:
            logger.error("--cache-dir required for --mode continuation. Run generate_train_cache.py first.")
            sys.exit(1)
        logger.info(f"Loading teacher continuation cache from {cache_dir}...")
        data_iter = create_continuation_data_iterator(cache_dir, seed=seed)
    else:
        logger.info("Starting FineWeb data stream...")
        data_iter = create_data_iterator(tokenizer, seq_len, seed=seed)

    # ── Training loop ─────────────────────────────────────────────────
    logger.info(f"Starting training from step {start_step} (mode={mode})...")
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_count = 0
    t0 = time.time()

    config = {
        "student_repo": student_repo, "teacher_model": teacher_model,
        "num_steps": num_steps, "batch_size": batch_size, "grad_accum": grad_accum,
        "lr": lr, "warmup_steps": warmup_steps, "seq_len": seq_len,
        "mode": mode, "cache_dir": cache_dir, "seed": seed,
    }
    (output_path / "train_config.json").write_text(json.dumps(config, indent=2))

    for step in range(start_step, num_steps):
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        for micro in range(grad_accum):
            if mode == "continuation":
                full_ids, prompt_len = next(data_iter)
                input_ids = full_ids.to(device)
            else:
                try:
                    input_ids = next(data_iter).to(device)
                except StopIteration:
                    data_iter = create_data_iterator(tokenizer, seq_len, seed=seed + step)
                    input_ids = next(data_iter).to(device)
                prompt_len = None

            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits

            student_logits = student(input_ids).logits

            if mode == "continuation" and prompt_len is not None:
                loss = compute_continuation_kl_loss(teacher_logits, student_logits, prompt_len) / grad_accum
            else:
                loss = compute_kl_loss(teacher_logits, student_logits) / grad_accum
            loss.backward()

            accum_loss += loss.item() * grad_accum
            accum_count += 1

            del teacher_logits, student_logits, input_ids

        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if (step + 1) % log_every == 0:
            avg_loss = accum_loss / max(accum_count, 1)
            elapsed = time.time() - t0
            steps_per_sec = (step + 1 - start_step) / elapsed
            eta_min = (num_steps - step - 1) / max(steps_per_sec, 1e-6) / 60
            logger.info(
                f"Step {step + 1}/{num_steps} | "
                f"KL loss={avg_loss:.6f} | "
                f"LR={current_lr:.2e} | "
                f"{steps_per_sec:.2f} steps/s | "
                f"ETA={eta_min:.0f}min"
            )
            accum_loss = 0.0
            accum_count = 0

        # Checkpoint
        if (step + 1) % save_every == 0:
            save_checkpoint(student, optimizer, step + 1, output_dir)

    # ── Final save ────────────────────────────────────────────────────
    final_dir = output_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model saved to {final_dir}")
    logger.info(f"Next: python scripts/eval_local.py --student-repo {final_dir}")


if __name__ == "__main__":
    main()
