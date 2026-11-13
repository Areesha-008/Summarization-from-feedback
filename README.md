# Summarization from Human Feedback

Reproducing the RLHF pipeline of *Learning to Summarize from Human Feedback* (Stiennon et al., 2020)
with `HuggingFaceTB/SmolLM2-135M`: SFT -> reward model -> PPO, and DPO as an alternative to PPO.
All losses (SFT, Bradley-Terry, PPO, DPO) are implemented from scratch in PyTorch.

## Repository layout

```
.
├── starter/              # given data / model loading code (unchanged)
│   ├── config.py         #   fixed settings: model, datasets, sizes, seed
│   ├── models.py         #   load_tokenizer, load_causal_lm
│   ├── data.py           #   load, clean, filter, tokenise
│   ├── batching.py       #   collate_sft / collate_pref / collate_prompts
│   └── check_starter.py  #   Task 1.1 sanity check
├── common.py             # shared helpers + project paths: token_logprobs, summary mask, generation
├── train_sft.py          # Task 2 - Stage 1: supervised fine-tuning
├── train_reward.py       # Task 3 - Stage 2: reward model
├── train_ppo.py          # Task 4 - Stage 3: PPO
├── train_dpo.py          # Task 5 - Stage 4: DPO
├── evaluate.py           # Task 6 - compare SFT / PPO / DPO on 300 eval posts
├── app/                  # Task 8 - web front end
├── notebooks/            # Colab runners (GPU)
├── results/
│   ├── logs/             #   per-stage training logs (csv)
│   ├── figures/          #   plots used in the report / blog
│   └── *.md, *.jsonl, *.csv   tables, eval summaries, human eval, analysis
├── checkpoints/          # (git-ignored) sft/, reward/, ppo/, dpo/
└── data_cache/           # (git-ignored) pickled tokenised splits
```

Every script is run from the repository root; paths are resolved through `common.py`,
so outputs always land in the folders above.

## Setup

```
pip install -r requirements.txt
cd starter && python check_starter.py && cd ..
```

On Colab, open `notebooks/colab_runner.ipynb` and use *Runtime → Run all*: it runs every stage below in
order, resumes after a disconnect, and stores `checkpoints/`, `data_cache/` and `results/` on Google Drive.

## How to run

| Step | Command | Produces |
|------|---------|----------|
| Task 1 - data table + mask check | `python common.py` | `results/table1_data.md`, `results/logs/data_summary.json` |
| Task 2 - SFT | `python train_sft.py` | `checkpoints/sft/`, `results/logs/sft_log.csv`, `results/figures/fig1_sft_loss.png`, `results/sft_samples.md` |
| Task 3 - reward model | `python train_reward.py` | `checkpoints/reward/reward_model.pt`, `results/logs/reward_log.csv`, `results/logs/reward_val_scores.jsonl`, `results/figures/fig2_reward.png` |
| Task 4 - PPO | `python train_ppo.py` | `checkpoints/ppo/`, `results/logs/ppo_log.csv`, `results/figures/fig3_ppo.png` |
| Task 5 - DPO | `python train_dpo.py` | `checkpoints/dpo/`, `results/logs/dpo_log.csv`, `results/figures/fig4_dpo.png` |
| Task 6 - evaluation | `python evaluate.py` | `results/eval_summaries.jsonl`, `results/table3_final.md`, `results/table2_training.md`, `results/correctness_checks.md`, `results/human_eval.csv`, skeletons of `results/samples.md` and `results/analysis.md` |
| Section 3.2 - human evaluation | fill `results/human_eval.csv`, then `python evaluate.py --tally-human` | `results/table4_human_eval.md` |
| Task 8 - front end | TODO | |

Stages must run in order: reward, PPO and DPO all start from `checkpoints/sft/`, and PPO also needs
`checkpoints/reward/`.
Every stage loads models through `common.load_lm`, which casts to float32 (SmolLM2 is stored in
bfloat16, and `transformers` >= 5 would otherwise train in bfloat16, where small AdamW updates are lost).
Each script prints `model dtype torch.float32` at start-up, and writes its final numbers and correctness
checks to `results/logs/<stage>_summary.json`, which `evaluate.py` turns into Table 2 and the checks below.

## Checkpoints

TODO (Hugging Face / Google Drive links)

## Correctness checks

TODO: paste `results/correctness_checks.md` here after `python evaluate.py`.

## Front end

TODO (screenshot)
