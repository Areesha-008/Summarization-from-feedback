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

from common import generate_summaries, load_data, token_logprobs
from batching import collate_sft  # path already set by common
from config import SEED
from models import load_causal_lm, load_tokenizer


LR = 5e-5
BATCH = 16
LOG_EVERY = 250
MAX_NEW = 48
OUT = "checkpoints/sft"

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



def write_samples(path, posts, before, after):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, post in enumerate(posts):
            f.write(f"### Post {i}\n")
            f.write(f"{post.prompt[-400:]}\n\n")
            f.write(f"Reference:\n{post.summary}\n\n")
            f.write(f"Base:\n{before[i]}\n\n")
            f.write(f"SFT:\n{after[i]}\n\n")
            f.write("-" * 40 + "\n\n")


def main():
    torch.manual_seed(SEED)
    tok = load_tokenizer()
    sft_train, sft_val, _, _, eval_posts = load_data(tok)
    model = load_causal_lm() 

    sample_posts = eval_posts[:5]
    before = generate_summaries(model, tok, sample_posts, MAX_NEW)

    collate = partial(collate_sft, pad_id=tok.pad_token_id)
    train_loader = DataLoader(sft_train, BATCH, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(sft_val, BATCH, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    log = []      # rows of (step, train_loss, val_loss)
    running = []  # train losses since last log

    log.append((0, None, evaluate(model, val_loader)))
    t0 = time.time()
    model.train()

    for step, batch in enumerate(train_loader, start=1):
        loss = sft_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        running.append(loss.item())
        if step % LOG_EVERY == 0 or step == len(train_loader):
            train_mean = sum(running) / len(running)
            val = evaluate(model, val_loader)
            log.append((step, train_mean, val))
            running = []
            elapsed = (time.time() - t0) / 60
            print(f"step {step}  train {train_mean:.4f}  val {val:.4f}  {elapsed:.2f} min")

    train_minutes = (time.time() - t0) / 60

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    after = generate_summaries(model, tok, sample_posts, MAX_NEW)

    os.makedirs("results", exist_ok=True)
    with open("results/sft_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])
        writer.writerows(log)

    write_samples("results/sft_samples.md", sample_posts, before, after)

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
    plt.savefig("results/fig1_sft_loss.png", dpi=150)
    plt.close()

    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss:   {val_losses[-1]:.4f}")
    print(f"Train Minutes:    {train_minutes:.2f}")
    print(f"Device:           {gpu_name}")


if __name__ == "__main__":
    main()