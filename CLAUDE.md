# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Distil (SN97) — a Bittensor subnet for competitive model distillation of Qwen3.5-35B-A3B. Miners distill the teacher into smaller models (≤5.25B params), validators compute full-distribution KL divergence, and the best miner (lowest KL) gets 100% of emissions (winner-take-all).

## Commands

```bash
# Install
pip install -e .                          # editable install
pip install -e ".[dev]"                   # with pytest

# Test (no GPU or chain required)
python -m pytest sim/test_full.py         # all tests
python -m pytest sim/test_full.py -k test_kl_divergence_properties  # single test

# Run validator (requires 80GB+ VRAM)
python validator.py --network finney --netuid 97 --wallet-name W --hotkey-name H

# Run miner (one-shot commitment, permanent)
python miner.py --network finney --netuid 97 --wallet-name W --hotkey-name H --model-repo user/model

# API server
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

No linter or formatter is configured.

## Architecture

**Entry points** — top-level scripts, not inside packages:
- `validator.py` — main production validator loop (loads teacher on GPU, evaluates students each epoch, sets weights on-chain)
- `miner.py` — one-shot model commitment script (validates architecture then commits HF repo link on-chain)
- `scripts/remote_validator.py` — king-of-the-hill distributed validator (Hetzner chain node + Lium GPU pod)
- `scripts/pod_eval.py` — GPU evaluation runner for remote pods (no chain access)

**`eval/`** — core evaluation logic, all stateless functions:
- `model_checker.py` — architecture validation (param count, vocab size, quantization rejection), SHA256 hash computation, copy detection, tokenizer verification, MoE-aware param counting
- `kl_divergence.py` — full-distribution KL(teacher||student) on GPU across all 248K vocab tokens. Key optimization: teacher continuations are pre-generated once per epoch and cached, then reused for all students
- `dataset.py` — FineWeb prompt loading and block-seeded deterministic sampling (miners can't predict prompts)
- `scoring.py` — EMA score smoothing, winner-take-all weight computation, failure/staleness tracking, state persistence (JSON files in `state/`)

**`api/`** — FastAPI dashboard backend with in-memory + disk caching, serves metagraph/scores/commitments

**`sim/`** — offline simulation tests (`test_full.py`) that mock chain and GPU interactions

## Key design details

- **Winner-take-all**: only the single best KL score gets weight=1.0; everyone else gets 0.0
- **One commitment per hotkey, permanent**: miners cannot update their model after committing
- **King-of-the-hill**: only new/unevaluated challengers are evaluated against the current king; 1% epsilon threshold prevents noisy flipping
- **Copy detection**: SHA256 of first safetensors shard; first committer owns the hash, duplicates are permanently disqualified
- **Teacher continuation caching**: teacher generates continuations once per epoch (expensive), all students evaluated against same cached continuations
- **State persistence**: all validator state lives in `state/*.json` files (scores, failures, hashes, disqualified UIDs, commitment cache)
- **EMA smoothing**: `alpha=0.3` blends new KL scores with historical; staleness threshold at 3 consecutive failures
