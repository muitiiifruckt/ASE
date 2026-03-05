# ASE: Adaptive Self-improvement via RLHF for Code Generation

This project explores Reinforcement Learning from Human Feedback (RLHF) techniques to improve code generation quality using the **DeepSeek-Coder-1.3B-Instruct** model fine-tuned on the **MBPP** (Mostly Basic Python Problems) benchmark.

Two RL algorithms are compared:

| Algorithm | Notebook | Description |
|-----------|----------|-------------|
| **GRPO** | `grpo-asem.ipynb` | Group Relative Policy Optimization via TRL library |
| **PF-PPO** | `pf-ppo-asem.ipynb` | Custom Pairwise Feedback PPO (Algorithm 1 from the paper) |

---

## Project Overview

### Base Model
- `deepseek-ai/deepseek-coder-1.3b-instruct`

### Dataset
- **MBPP** (`mbpp.jsonl`) — Mostly Basic Python Problems
  - Train: 350 examples
  - Validation: 50 examples
  - Test: 100 examples
  - Split: `random.seed(42)` shuffle, then 350/50/100

### Reward Signals

| Signal | Description |
|--------|-------------|
| **R_test** | Unit-test reward — runs generated code against assert-based tests, returns `passed / total` |
| **R_LLM** | LLM-judge reward — uses `flowaicom/Flow-Judge-v0.1` (Phi-3 based) to score code quality in `[0, 1]` |
| **R_static** | Static analysis proxy — measures cyclomatic complexity, nesting depth, and LOC |

### Final Evaluation Metric
The final score is the harmonic mean of test pass rate and static quality:

$$R_{final} = \frac{2 \cdot R_{test} \cdot R_{static}}{R_{test} + R_{static}}$$

---

## Algorithm Details

### GRPO (`grpo-asem.ipynb`)
Uses the [TRL](https://github.com/huggingface/trl) library `GRPOTrainer` with the following config:
- `learning_rate = 3e-5`
- `num_generations = 4` (K samples per problem)
- `max_steps = 300`
- `max_new_tokens = 256`, `temperature = 0.7`, `top_p = 0.9`
- Trains with both `reward_fn_test` and `reward_llm_grpo_fn`

### PF-PPO (`pf-ppo-asem.ipynb`)
Custom implementation of **Pairwise Feedback PPO** (Algorithm 1):
1. For each problem, generate **K=4** solutions
2. Score each solution with the reward function
3. Form preference pairs `(better, worse)` where `|R_i - R_j| > δ`
4. For each pair compute:
   - **Policy loss** with PPO clipping on trajectory-level log-prob ratio
   - **Value loss** (MSE of critic `V_φ` vs actual rewards)
   - **KL penalty** against a frozen reference model
5. Update both policy `π_θ` (AdamW) and value head `V_φ` (AdamW)

Key hyperparameters:
- `learning_rate = 3e-5`, `clip_range = 0.2`, `kl_coef = 0.01`
- `K = 4`, `delta = -0.1`, `max_steps = 100`
- Value head: 2-layer MLP (2048 → 1024 → 1), ~2.1M parameters, `bfloat16`

---

## Repository Structure

```
ASE/
├── grpo-asem.ipynb       # GRPO training (TRL-based)
├── pf-ppo-asem.ipynb     # Custom PF-PPO training from scratch
└── README.md
```

---

## Requirements

The notebooks are designed to run on **Kaggle** (GPU T4/P100) or Google Colab. Key dependencies:

```
torch >= 2.0
transformers >= 4.56
trl >= 0.26
accelerate >= 1.4
datasets >= 3.0
```

Dataset file must be placed at `/kaggle/input/asemjson/mbpp.jsonl`.

---

## Running

1. Open either notebook in Kaggle or Colab with GPU enabled
2. Ensure `mbpp.jsonl` is available at the expected path
3. Run all cells sequentially

