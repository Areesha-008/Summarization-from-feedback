"""Task 3 - Stage 2: reward model.

RewardModel = SFT transformer body (model.model) + nn.Linear(hidden, 1) on the last
non-padding token. Bradley-Terry loss. 1 epoch, AdamW lr 1e-5, batch 8 pairs.
Report pairwise accuracy on 1,000 val pairs. Save to checkpoints/reward/.

Outputs: checkpoints/reward/reward_model.pt, results/logs/reward_log.csv,
results/logs/reward_val_scores.jsonl (for the Task 7.1 error analysis),
results/figures/fig2_reward.png.
Progress is saved to checkpoints/reward_resume/ every 250 steps; rerunning resumes from there.

Other stages load the trained model with `load_reward_model()`.
"""
import csv
import json
import os
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR, find_resume, load_data, load_lm,
                    load_resume_state, save_resume, write_stage_summary)
from batching import collate_pref
from config import SEED
from models import load_tokenizer

LR = 1e-5
BATCH = 8
LOG_EVERY = 250
SFT_CKPT = os.path.join(CHECKPOINTS_DIR, "sft")
OUT = os.path.join(CHECKPOINTS_DIR, "reward")
RESUME = os.path.join(CHECKPOINTS_DIR, "reward_resume")
WEIGHTS = "reward_model.pt"


class RewardModel(nn.Module):
    """r_phi(x, y): one scalar per (prompt + summary) sequence."""

    def __init__(self, body):
        super().__init__()
        self.body = body                                      # transformer without the LM head
        self.head = nn.Linear(body.config.hidden_size, 1)     # hidden state -> scalar reward

    @property
    def device(self):
        return self.head.weight.device

    def forward(self, input_ids, attention_mask):
        hidden = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Rows are right-padded and pad == eos, so the last real token is found from the
        # attention mask (number of real tokens - 1), never from pad_token_id.
        last = attention_mask.sum(dim=1) - 1
        h_last = hidden[torch.arange(hidden.size(0), device=hidden.device), last]  # (B, H)
        return self.head(h_last).squeeze(-1)                                        # (B,)

    def save_pretrained(self, path):
        """Body + head in one state dict (also used by common.save_resume)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, WEIGHTS))


def build_reward_model(sft_path=None):
    """Fresh reward model: SFT body + new linear head."""
    body = load_lm(sft_path or SFT_CKPT).model     # model.model = transformer body; lm_head is dropped
    return RewardModel(body).to(body.device)


def load_reward_model(path=None, sft_path=None, frozen=True):
    """Trained reward model. The SFT checkpoint only provides the architecture/config."""
    model = build_reward_model(sft_path)
    state = torch.load(os.path.join(path or OUT, WEIGHTS), map_location=model.device, weights_only=True)
    model.load_state_dict(state)
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def pair_rewards(model, batch):
    """(r(x, y_chosen), r(x, y_rejected)), each shape (B,)."""
    device = model.device
    r_chosen = model(batch["chosen_ids"].to(device), batch["chosen_mask"].to(device))
    r_rejected = model(batch["rejected_ids"].to(device), batch["rejected_mask"].to(device))
    return r_chosen, r_rejected


def bradley_terry_loss(r_chosen, r_rejected):
    """-E[log sigma(r(x, y_w) - r(x, y_l))]."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


@torch.no_grad()
def evaluate(model, loader):
    """Pairwise accuracy: share of pairs with r(x, y_w) > r(x, y_l). Also returns the scores."""
    was_training = model.training
    model.eval()
    chosen, rejected = [], []
    for batch in loader:
        r_c, r_r = pair_rewards(model, batch)
        chosen += r_c.tolist()
        rejected += r_r.tolist()
    model.train(was_training)
    acc = sum(c > r for c, r in zip(chosen, rejected)) / len(chosen)
    return acc, chosen, rejected


def write_log(log, path=os.path.join(LOGS_DIR, "reward_log.csv")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_pairwise_acc"])
        writer.writerows(log)


def write_val_scores(pairs, chosen, rejected, path=os.path.join(LOGS_DIR, "reward_val_scores.jsonl")):
    """Per-pair scores on the validation set, to find the pairs the model gets wrong (Task 7.1)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex, r_c, r_r in zip(pairs, chosen, rejected):
            f.write(json.dumps({"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"],
                                "r_chosen": r_c, "r_rejected": r_r, "correct": r_c > r_r}) + "\n")


def plot(log, path=os.path.join(FIGURES_DIR, "fig2_reward.png")):
    steps = [row[0] for row in log if row[1] is not None]
    losses = [row[1] for row in log if row[1] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, losses, marker="o")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("Bradley-Terry loss (train)")
    axes[1].plot([row[0] for row in log], [row[2] for row in log], marker="s")
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("pairwise accuracy (val)")
    axes[1].legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if not os.path.isfile(os.path.join(SFT_CKPT, "config.json")):
        raise SystemExit(f"No SFT checkpoint at {SFT_CKPT}. Run train_sft.py first.")
    torch.manual_seed(SEED)                 # also fixes the new head's initialisation
    tok = load_tokenizer()
    _, _, pref_train, pref_val, _ = load_data(tok)

    model = build_reward_model()
    print(f"model dtype {model.head.weight.dtype}", flush=True)
    resume_from = find_resume(RESUME)
    if resume_from:
        model.load_state_dict(torch.load(os.path.join(resume_from, WEIGHTS),
                                         map_location=model.device, weights_only=True))

    collate = partial(collate_pref, pad_id=tok.pad_token_id)
    order = torch.Generator()
    order.manual_seed(SEED)                 # same shuffle every run, so a resumed run skips the right batches
    train_loader = DataLoader(pref_train, BATCH, shuffle=True, collate_fn=collate, generator=order)
    val_loader = DataLoader(pref_val, BATCH, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    if resume_from:
        state = load_resume_state(resume_from, opt)
        start_step, log, elapsed_done = state["step"], state["log"], state["elapsed_min"]
        print(f"resuming from step {start_step}", flush=True)
    else:
        start_step, elapsed_done = 0, 0.0
        log = [(0, None, evaluate(model, val_loader)[0])]
        print(f"step 0  val acc {log[0][2]:.4f}  (random head, expect ~0.5)", flush=True)

    running = []
    t0 = time.time()
    model.train()
    n_steps = len(train_loader)

    for step, batch in enumerate(train_loader, start=1):
        if step <= start_step:
            continue
        r_chosen, r_rejected = pair_rewards(model, batch)
        loss = bradley_terry_loss(r_chosen, r_rejected)
        loss.backward()
        opt.step()
        opt.zero_grad()

        running.append(loss.item())
        if step % LOG_EVERY == 0 or step == n_steps:
            train_loss = sum(running) / len(running)
            val_acc = evaluate(model, val_loader)[0]
            log.append((step, train_loss, val_acc))
            running = []
            elapsed = elapsed_done + (time.time() - t0) / 60
            print(f"step {step}/{n_steps}  train {train_loss:.4f}  val acc {val_acc:.4f}  "
                  f"{elapsed:.1f} min", flush=True)
            save_resume(RESUME, model, tok, opt, {"step": step, "log": log, "elapsed_min": elapsed})
            write_log(log)

    train_minutes = elapsed_done + (time.time() - t0) / 60
    model.save_pretrained(OUT)

    val_acc, chosen, rejected = evaluate(model, val_loader)
    write_val_scores(pref_val, chosen, rejected)
    write_log(log)
    plot(log)

    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"final train {log[-1][1]:.4f}  val pairwise acc {val_acc:.4f}", flush=True)
    print(f"train minutes {train_minutes:.2f}  device {gpu_name}", flush=True)
    print(f"saved {OUT}", flush=True)
    write_stage_summary("reward", {"final_train_loss": log[-1][1], "val_pairwise_acc": val_acc,
                                   "minutes": train_minutes})


if __name__ == "__main__":
    main()
