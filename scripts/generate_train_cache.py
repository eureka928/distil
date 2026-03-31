#!/usr/bin/env python3
"""
Pre-generate teacher continuations for Phase 2 continuation-only KL training.

Matches the validator's exact generation settings:
  - FineWeb prompts via eval/dataset.py
  - Teacher generates 512 tokens with temperature=0.7, top_p=0.9
  - Saves only token IDs + prompt_len (teacher logits recomputed during training)

Usage:
    python scripts/generate_train_cache.py --num-prompts 60 --num-blocks 500
    python scripts/generate_train_cache.py --num-prompts 60 --num-blocks 100 --output-dir ./state/train_cache
"""
import sys
import logging
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.model_checker import TEACHER_MODEL
from eval.dataset import sample_prompts_from_dataset, format_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_train_cache")

MAX_NEW_TOKENS = 512
MAX_PROMPT_TOKENS = 1024


@click.command()
@click.option("--teacher-model", default=TEACHER_MODEL)
@click.option("--num-prompts", type=int, default=40, help="Prompts to sample per block (matches validator)")
@click.option("--num-blocks", type=int, default=500, help="Number of simulated blocks for diversity")
@click.option("--output-dir", type=click.Path(), default="state/train_cache")
@click.option("--device", default="cuda")
@click.option("--teacher-precision", type=click.Choice(["bf16", "4bit"]), default="4bit",
              help="Teacher precision: bf16 (B200/H100) or 4bit (A100)")
def main(teacher_model, num_prompts, num_blocks, output_dir, device, teacher_precision):
    """Generate teacher continuation cache for Phase 2 training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("*.pt"))
    if existing:
        logger.info(f"Found {len(existing)} existing cached samples in {output_path}")

    logger.info(f"Generating {num_prompts} prompts x {num_blocks} blocks = {num_prompts * num_blocks} samples")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)

    # Load teacher
    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    if teacher_precision == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to bf16")
            load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    logger.info(f"Loading teacher ({teacher_precision})...")

    teacher = AutoModelForCausalLM.from_pretrained(teacher_model, **load_kwargs)
    teacher.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Teacher loaded: {used:.1f}GB VRAM")

    total_generated = 0

    for block_idx in range(num_blocks):
        # Simulate different block numbers for diverse prompt sampling
        block_number = block_idx * 1000 + 42
        raw_prompts = sample_prompts_from_dataset(num_prompts, block_number=block_number)
        prompt_texts = [format_prompt(p) for p in raw_prompts]
        prompt_texts = [p for p in prompt_texts if p]

        block_generated = 0
        block_skipped = 0
        for i, text in enumerate(prompt_texts):
            sample_id = f"block{block_number}_p{i}"
            sample_path = output_path / f"{sample_id}.pt"

            if sample_path.exists():
                block_skipped += 1
                continue

            input_ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS,
            ).input_ids.to(device)
            prompt_len = input_ids.shape[1]

            torch.manual_seed(block_number)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(block_number)

            with torch.no_grad():
                output = teacher.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            gen_len = output.shape[1] - prompt_len

            if gen_len == 0:
                continue

            torch.save({
                "full_ids": output.cpu(),
                "prompt_len": prompt_len,
                "gen_len": gen_len,
            }, sample_path)

            block_generated += 1

        total_generated += block_generated
        logger.info(
            f"Block {block_idx + 1}/{num_blocks}: "
            f"{block_generated} new, {block_skipped} cached"
        )

    logger.info(f"Done. {total_generated} new samples. Total in cache: {len(list(output_path.glob('*.pt')))}")


if __name__ == "__main__":
    main()
