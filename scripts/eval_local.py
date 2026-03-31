#!/usr/bin/env python3
"""
Local KL evaluation — mirrors the validator's exact scoring pipeline.

Use this to test your student model BEFORE making the irreversible on-chain commit.

Two-pass approach for 1x A100 80GB:
  Pass 1: Load teacher bf16, generate continuations, cache logits to disk
  Pass 2: Load student bf16, compute KL against cached teacher logits

Teacher logit cache is reusable across students (keyed by seed + num_prompts).

Usage:
    python scripts/eval_local.py --student-repo user/my-model
    python scripts/eval_local.py --student-repo ./checkpoints/run1/final --num-prompts 40
    python scripts/eval_local.py --student-repo user/my-model --king-kl 0.063
"""
import sys
import gc
import logging
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.kl_divergence import compute_kl_from_logits
from eval.dataset import sample_prompts_from_dataset, format_prompt
from eval.model_checker import check_model_architecture, verify_tokenizer_match, TEACHER_MODEL
from validator import model_sanity_check

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_local")

TEACHER_TOTAL_PARAMS_B = 35.0
DEFAULT_MAX_PARAM_RATIO = 0.15
MAX_NEW_TOKENS = 512
MAX_PROMPT_TOKENS = 1024
CACHE_DIR = Path("state/teacher_cache")


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cache_key(seed: int, num_prompts: int) -> str:
    return f"seed{seed}_n{num_prompts}"


def generate_teacher_cache(
    teacher_model_name: str,
    prompts: list[str],
    seed: int,
    device: str,
) -> Path:
    """Load teacher bf16, generate continuations, save logits to disk, unload."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = cache_key(seed, len(prompts))
    cache_path = CACHE_DIR / key
    cache_path.mkdir(parents=True, exist_ok=True)

    expected_files = [cache_path / f"prompt_{i}.pt" for i in range(len(prompts))]
    if all(f.exists() for f in expected_files):
        logger.info(f"Teacher cache found: {cache_path}")
        return cache_path

    logger.info(f"Generating teacher logit cache ({len(prompts)} prompts)...")
    logger.info("Loading teacher model (bf16) — this needs ~72GB VRAM...")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Teacher loaded: {used:.1f}GB VRAM used")

    for i, text in enumerate(prompts):
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS,
        ).input_ids.to(device)
        prompt_len = input_ids.shape[1]

        # Generate continuation (matching validator: seeded sampling)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

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
            torch.save({
                "full_ids": output.cpu(),
                "teacher_logits": None,
                "prompt_len": prompt_len,
                "gen_len": 0,
            }, cache_path / f"prompt_{i}.pt")
            logger.info(f"  Prompt {i}: {prompt_len} tokens, gen_len=0 (skipped)")
            continue

        with torch.no_grad():
            teacher_logits_full = teacher(output).logits

        # float16 halves cache size (~250MB/prompt) with negligible KL error
        teacher_cont_logits = teacher_logits_full[:, prompt_len - 1:-1, :].half().cpu()

        torch.save({
            "full_ids": output.cpu(),
            "teacher_logits": teacher_cont_logits,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
        }, cache_path / f"prompt_{i}.pt")

        size_mb = teacher_cont_logits.nelement() * 2 / 1e6
        logger.info(f"  Prompt {i}: {prompt_len}+{gen_len} tokens, logits={size_mb:.0f}MB")

    del teacher
    free_gpu()
    logger.info("Teacher unloaded, VRAM freed")

    return cache_path


def evaluate_student(
    student_repo: str,
    revision: str,
    cache_path: Path,
    num_prompts: int,
    device: str,
) -> dict:
    """Load student bf16, evaluate KL against cached teacher logits."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading student: {student_repo}")
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL, trust_remote_code=True,
    )
    student = AutoModelForCausalLM.from_pretrained(
        student_repo,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    student.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Student loaded: {used:.1f}GB VRAM used")

    sane, sane_reason = model_sanity_check(student, tokenizer, device)
    if not sane:
        return {"error": f"Sanity check failed: {sane_reason}"}

    # Evaluate KL per prompt
    kl_results = []
    for i in range(num_prompts):
        cached = torch.load(cache_path / f"prompt_{i}.pt", weights_only=True)
        prompt_len = cached["prompt_len"]
        gen_len = cached["gen_len"]

        if gen_len == 0 or cached["teacher_logits"] is None:
            continue

        full_ids = cached["full_ids"].to(device)

        with torch.no_grad():
            student_logits_full = student(full_ids).logits

        student_cont_logits = student_logits_full[:, prompt_len - 1:-1, :].float()
        teacher_cont_logits = cached["teacher_logits"].float().to(device)

        result = compute_kl_from_logits(teacher_cont_logits, student_cont_logits)
        result["prompt_len"] = prompt_len
        result["gen_len"] = gen_len
        kl_results.append(result)

        logger.debug(f"  Prompt {i}: KL={result['kl_mean']:.6f} ({result['n_positions']} positions)")

        del student_logits_full, student_cont_logits, teacher_cont_logits, full_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del student
    free_gpu()

    if not kl_results:
        return {"error": "No prompts evaluated (all had gen_len=0)"}

    total_positions = sum(r["n_positions"] for r in kl_results)
    avg_kl = sum(r["kl_mean"] * r["n_positions"] for r in kl_results) / total_positions
    all_means = [r["kl_mean"] for r in kl_results]
    kl_std = torch.tensor(all_means).std().item() if len(all_means) > 1 else 0.0
    kl_min = min(all_means)
    kl_max = max(all_means)

    return {
        "kl_mean": avg_kl,
        "kl_std": kl_std,
        "kl_min": kl_min,
        "kl_max": kl_max,
        "n_prompts": len(kl_results),
        "total_positions": total_positions,
    }


@click.command()
@click.option("--student-repo", required=True, help="HuggingFace repo or local path to student model")
@click.option("--revision", default="main", help="Model revision/commit SHA")
@click.option("--num-prompts", type=int, default=60, help="Number of FineWeb prompts (validator uses 60)")
@click.option("--seed", type=int, default=42, help="Seed for prompt selection and teacher generation (simulates block number)")
@click.option("--teacher-model", default=TEACHER_MODEL, help="Teacher model repo")
@click.option("--king-kl", type=float, default=None, help="Current king's KL for comparison")
@click.option("--skip-prechecks", is_flag=True, help="Skip architecture/tokenizer pre-checks")
@click.option("--device", default="cuda", help="Device for inference")
def main(student_repo, revision, num_prompts, seed, teacher_model, king_kl, skip_prechecks, device):
    """Evaluate a student model's KL-divergence against the teacher, matching validator scoring."""

    max_student_params_b = TEACHER_TOTAL_PARAMS_B * DEFAULT_MAX_PARAM_RATIO

    # ── Pre-checks ────────────────────────────────────────────────────
    if not skip_prechecks:
        logger.info("Running pre-checks...")

        # Architecture
        arch = check_model_architecture(student_repo, revision, max_student_params_b)
        if not arch["pass"]:
            logger.error(f"FAIL architecture: {arch['reason']}")
            sys.exit(1)
        params_b = arch.get("params_b", 0)
        logger.info(f"  params={params_b:.2f}B  vocab={arch.get('vocab_size', '?')}  "
                     f"moe={arch.get('is_moe', False)}  "
                     f"hybrid={arch.get('has_hybrid_attention', False)}")

        # Tokenizer
        try:
            tok_result = verify_tokenizer_match(student_repo, revision)
            if not tok_result["match"]:
                logger.error(f"FAIL tokenizer: {tok_result['reason']}")
                sys.exit(1)
            logger.info("  tokenizer: match")
        except Exception as e:
            logger.warning(f"  tokenizer check failed: {e} (continuing)")

        logger.info("Pre-checks passed")
    else:
        logger.info("Pre-checks skipped (--skip-prechecks)")

    # ── Load prompts (matches validator: full FineWeb, block-seeded) ──
    logger.info(f"Sampling {num_prompts} prompts from FineWeb (seed={seed})...")
    raw_prompts = sample_prompts_from_dataset(num_prompts, block_number=seed)
    prompt_texts = [format_prompt(p) for p in raw_prompts]
    prompt_texts = [p for p in prompt_texts if p]  # drop empty after sanitization
    logger.info(f"Got {len(prompt_texts)} prompts")

    # ── Teacher logit cache ───────────────────────────────────────────
    cache_path = generate_teacher_cache(teacher_model, prompt_texts, seed, device)

    # ── Student evaluation ────────────────────────────────────────────
    result = evaluate_student(student_repo, revision, cache_path, len(prompt_texts), device)

    if "error" in result:
        logger.error(f"Evaluation failed: {result['error']}")
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Student:     {student_repo}")
    print(f"KL mean:     {result['kl_mean']:.6f}")
    print(f"KL std:      {result['kl_std']:.6f}")
    print(f"KL range:    [{result['kl_min']:.6f}, {result['kl_max']:.6f}]")
    print(f"Prompts:     {result['n_prompts']}/{len(prompt_texts)} evaluated "
          f"({result['total_positions']} total positions)")

    if king_kl is not None:
        epsilon = 0.01
        threshold = king_kl * (1.0 - epsilon)
        if result["kl_mean"] < threshold:
            improvement = (1 - result["kl_mean"] / king_kl) * 100
            print(f"Verdict:     WOULD BEAT king ({king_kl:.4f}) by {improvement:.1f}%")
        elif result["kl_mean"] < king_kl:
            gap = (1 - result["kl_mean"] / king_kl) * 100
            print(f"Verdict:     WITHIN EPSILON — {gap:.1f}% better but need >{epsilon*100}% to dethrone")
        else:
            gap = (result["kl_mean"] / king_kl - 1) * 100
            print(f"Verdict:     WOULD LOSE to king ({king_kl:.4f}) by {gap:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
