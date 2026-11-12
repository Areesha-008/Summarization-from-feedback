# Summarization from Human Feedback

Reproducing the RLHF pipeline of *Learning to Summarize from Human Feedback* (Stiennon et al., 2020)
with `HuggingFaceTB/SmolLM2-135M`: SFT -> reward model -> PPO, and DPO as an alternative to PPO.

## Setup

```
pip install -r requirements.txt
cd starter && python check_starter.py
```

## Layout

| Path | Purpose |
|------|---------|
| `starter/` | given data / model loading code, unchanged |
| `common.py` | `token_logprobs`, summary mask, generation helper |
| `train_sft.py` | Stage 1 - SFT |
| `train_reward.py` | Stage 2 - reward model |
| `train_ppo.py` | Stage 3 - PPO |
| `train_dpo.py` | Stage 4 - DPO |
| `evaluate.py` | compare the three policies |
| `results/` | logs, tables, figures, summaries, analysis |
| `app/` | web front end |

## How to run

TODO

## Checkpoints

TODO (Hugging Face / Google Drive links)

## Correctness checks

TODO

## Front end

TODO (screenshot)
