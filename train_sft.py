"""Task 2 - Stage 1: supervised fine-tuning.

Loss: mean NLL over summary tokens only. 1 epoch, AdamW lr 5e-5, batch 16, clip 1.0.
Log train/val loss every 250 steps. Save to checkpoints/sft/.
"""
import csv
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from common import (CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR, RESULTS_DIR, find_resume,
                    generate_summaries, load_data, load_lm, load_resume_state, save_resume,
                    token_logprobs)
from batching import collate_sft  # path already set by common
from config import SEED
from models import load_tokenizer


LR = 5e-5
BATCH = 16
LOG_EVERY = 250
MAX_NEW = 48
OUT = os.path.join(CHECKPOINTS_DIR, "sft")
RESUME = os.path.join(CHECKPOINTS_DIR, "sft_resume")

def sft_loss(model,batch):
    device = model.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    logprobs, mask = token_logprobs(model, input_ids, attention_mask, prompt_lens)
    return -(logprobs * mask).sum() / mask.sum()    


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    device=model.device
    total_nll=0.0
    total_tokens=0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        logprobs, mask = token_logprobs(model, input_ids, attention_mask, prompt_lens)
        total_nll += -(logprobs * mask).sum().item()
        total_tokens += mask.sum().item()

    model.train()
    return total_nll / max(total_tokens, 1)



def write_log(log, path=os.path.join(LOGS_DIR, "sft_log.csv")):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])
        writer.writerows(log)


def write_samples(path, posts, before, after):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, post in enumerate(posts):
            f.write(f"### Post {i}\n")
            f.write(f"{post['prompt'][-400:]}\n\n")
            f.write(f"Reference:\n{post['summary']}\n\n")
            f.write(f"Base:\n{before[i]}\n\n")
            f.write(f"SFT:\n{after[i]}\n\n")
            f.write("-" * 40 + "\n\n")


def main():
    torch.manual_seed(SEED)
    tok = load_tokenizer()
    sft_train, sft_val, _, _, eval_posts = load_data(tok)
    sample_posts = eval_posts[:5]

    resume_from = find_resume(RESUME)
    model = load_lm(resume_from)
    print(f"model dtype {model.dtype}", flush=True)

    collate = partial(collate_sft, pad_id=tok.pad_token_id)
    order = torch.Generator()
    order.manual_seed(SEED)
    train_loader = DataLoader(sft_train, BATCH, shuffle=True, collate_fn=collate, generator=order)
    val_loader = DataLoader(sft_val, BATCH, shuffle=False, collate_fn=collate)
    n_steps = len(train_loader)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    if resume_from:
        state = load_resume_state(resume_from, opt)
        start_step, log = state["step"], state["log"]
        before, elapsed_done = state["before"], state["elapsed_min"]
        print(f"resuming from step {start_step}", flush=True)
    else:
        start_step, log, elapsed_done = 0, [], 0.0
        before = generate_summaries(model, tok, sample_posts, MAX_NEW)
    running = []

    if start_step == 0:
        log.append((0, None, evaluate(model, val_loader)))
    t0 = time.time()
    model.train()
    if start_step:
        print(f"skipping {start_step} finished steps", flush=True)

    for step, batch in enumerate(train_loader, start=1):
        if step <= start_step:
            continue
        loss = sft_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        running.append(loss.item())
        if step % LOG_EVERY == 0 or step == n_steps:
            train_mean = sum(running) / len(running)
            val = evaluate(model, val_loader)
            log.append((step, train_mean, val))
            running = []
            elapsed = elapsed_done + (time.time() - t0) / 60
            print(f"step {step}  train {train_mean:.4f}  val {val:.4f}  {elapsed:.2f} min", flush=True)
            save_resume(RESUME, model, tok, opt,
                        {"step": step, "log": log, "before": before, "elapsed_min": elapsed})
            write_log(log)

    train_minutes = elapsed_done + (time.time() - t0) / 60

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    after = generate_summaries(model, tok, sample_posts, MAX_NEW)

    write_log(log)
    write_samples(os.path.join(RESULTS_DIR, "sft_samples.md"), sample_posts, before, after)

    # Plot training and validation curves
    train_steps = [row[0] for row in log if row[1] is not None]
    train_losses = [row[1] for row in log if row[1] is not None]
    val_steps = [row[0] for row in log]
    val_losses = [row[2] for row in log]

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="Train Loss", marker="o")
    plt.plot(val_steps, val_losses, label="Val Loss", marker="s")
    plt.xlabel("Step")
    plt.ylabel("NLL per summary token")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_sft_loss.png"), dpi=150)
    plt.close()

    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss:   {val_losses[-1]:.4f}")
    print(f"Train Minutes:    {train_minutes:.2f}")
    print(f"Device:           {gpu_name}")


if __name__ == "__main__":
    main()