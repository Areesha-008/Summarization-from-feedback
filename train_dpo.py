"""Task 5 - Stage 4: DPO.

Trainable policy + frozen reference, both from SFT. Sequence logprob = sum over summary
tokens. DPO loss with beta=0.1. 1 epoch, AdamW lr 2e-6, batch 8 pairs.
Log loss + implicit reward accuracy every 250 steps. Save to checkpoints/dpo/.

Outputs: checkpoints/dpo/, results/logs/dpo_log.csv, results/figures/fig4_dpo.png.
Correctness checks printed at start: reference frozen, first-step loss = log 2.
Progress is saved to checkpoints/dpo_resume/ every 250 steps; rerunning resumes from there.
"""
import csv
import math
import os
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR, find_resume, load_data,
                    load_lm, load_resume_state, save_resume, token_logprobs)
from batching import collate_pref
from config import SEED
from models import load_tokenizer

BETA = 0.1
LR = 2e-6
BATCH = 8
LOG_EVERY = 250
SFT_CKPT = os.path.join(CHECKPOINTS_DIR, "sft")
OUT = os.path.join(CHECKPOINTS_DIR, "dpo")
RESUME = os.path.join(CHECKPOINTS_DIR, "dpo_resume")


def sequence_logprob(model, input_ids, attention_mask, prompt_lens):
    """log π(y | x) for each row. Shape (batch)."""
    logprobs, mask = token_logprobs(model, input_ids, attention_mask, prompt_lens)
    return (logprobs * mask).sum(dim=-1)  # sum over summary 



def pair_margin(policy, ref, batch):
    """One margin per pair. Shape (batch,)."""
    device=policy.device
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)   
    pi_chosen = sequence_logprob(policy, chosen_ids, chosen_mask, prompt_lens)
    pi_rejected = sequence_logprob(policy, rejected_ids, rejected_mask, prompt_lens)
    with torch.no_grad():
        ref_chosen = sequence_logprob(ref, chosen_ids, chosen_mask, prompt_lens)
        ref_rejected = sequence_logprob(ref, rejected_ids, rejected_mask, prompt_lens)
    return BETA * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    


def dpo_loss(margin):
    return -F.logsigmoid(margin).mean()
    


@torch.no_grad()
def implicit_accuracy(policy, ref, loader):
    was_training = policy.training
    policy.eval()
    correct = 0.0
    total = 0.0
    for batch in loader:
        margin = pair_margin(policy, ref, batch)
        correct += (margin > 0).sum().item()
        total += margin.numel()
    policy.train(was_training)
    return correct / total


def write_log(log, path=os.path.join(LOGS_DIR, "dpo_log.csv")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_implicit_acc"])
        writer.writerows(log)


def main():
    if not os.path.isfile(os.path.join(SFT_CKPT, "config.json")):
        raise SystemExit(f"No SFT checkpoint at {SFT_CKPT}. Run train_sft.py first.")
    torch.manual_seed(SEED)
    tok = load_tokenizer()
    _, _, pref_train, pref_val, _ = load_data(tok)
    resume_from = find_resume(RESUME)
    policy = load_lm(resume_from or SFT_CKPT)
    ref = load_lm(SFT_CKPT)
    print(f"model dtype {policy.dtype}", flush=True)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    print("reference frozen:", all(not param.requires_grad for param in ref.parameters()))

    collate = partial(collate_pref, pad_id=tok.pad_token_id)
    order = torch.Generator()
    order.manual_seed(SEED)  # same shuffle on every run, so a resumed run skips the right batches
    train_loader = DataLoader(pref_train, BATCH, shuffle=True, collate_fn=collate, generator=order)
    val_loader = DataLoader(pref_val, BATCH, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)

    if resume_from:
        state = load_resume_state(resume_from, opt)
        start_step, log, elapsed_done = state["step"], state["log"], state["elapsed_min"]
        print(f"resuming from step {start_step}", flush=True)
    else:
        start_step, elapsed_done = 0, 0.0
        log = [(0, None, implicit_accuracy(policy, ref, val_loader))]
        print(f"step 0  val acc {log[0][2]:.4f}", flush=True)
    running = []
    t0 = time.time()
    policy.train()
    n_steps = len(train_loader)

    for step, batch in enumerate(train_loader, start=1):
        if step <= start_step:
            continue
        margin = pair_margin(policy, ref, batch)
        loss = dpo_loss(margin)
        if step == 1:
            print(f"first-step loss {loss.item():.4f}  expect {math.log(2):.4f}", flush=True)
        loss.backward()
        opt.step()
        opt.zero_grad()

        running.append(loss.item())
        if step % LOG_EVERY == 0 or step == n_steps:
            train_loss = sum(running) / len(running)
            val_acc = implicit_accuracy(policy, ref, val_loader)
            log.append((step, train_loss, val_acc))
            running = []
            elapsed = elapsed_done + (time.time() - t0) / 60
            print(
                f"step {step}/{n_steps}  train {train_loss:.4f}  val acc {val_acc:.4f}  "
                f"{elapsed:.1f} min",
                flush=True,
            )
            save_resume(RESUME, policy, tok, opt, {"step": step, "log": log, "elapsed_min": elapsed})
            write_log(log)

    train_minutes = elapsed_done + (time.time() - t0) / 60
    os.makedirs(OUT, exist_ok=True)
    policy.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    steps = [row[0] for row in log if row[1] is not None]
    losses = [row[1] for row in log if row[1] is not None]
    acc_steps = [row[0] for row in log]
    accs = [row[2] for row in log]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, losses)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("DPO loss")
    axes[1].plot(acc_steps, accs)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("implicit reward accuracy")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_dpo.png"), dpi=150)
    plt.close(fig)

    write_log(log)
    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"final train {log[-1][1]:.4f}  val acc {log[-1][2]:.4f}", flush=True)
    print(f"train minutes {train_minutes:.2f}  device {gpu_name}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
