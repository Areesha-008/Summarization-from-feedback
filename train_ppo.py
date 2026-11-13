"""Task 4 - Stage 3: PPO.

Policy + value head (from SFT), frozen reference (SFT), frozen reward model.
Rollout 16 prompts -> per-token KL-penalised reward -> returns/advantages (gamma=1)
-> K=4 PPO epochs (clipped policy loss + value loss + entropy bonus).
eps=0.2, beta_KL=0.05, c_V=0.1, c_H=0.01, AdamW lr 1e-5, clip 1.0, 200 iterations.
Save policy (no value head) to checkpoints/ppo/.

Each generated token is an action a_t; the state s_t is the prompt plus the tokens
generated so far. Scoring is done on right-padded prompt+summary rows (the layout SFT
and the reward model were trained on), so position t of a summary is predicted by the
hidden state at index prompt_len + t - 1 of its row.

Outputs: checkpoints/ppo/, results/logs/ppo_log.csv, results/figures/fig3_ppo.png.
Correctness checks printed: frozen models, KL = 0 at iteration 1, ratio = 1 and
clip fraction = 0 in the first epoch of every iteration.
Progress is saved to checkpoints/ppo_resume/ every SAVE_EVERY iterations.
"""
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR, decode_summary, find_resume,
                    generate_ids, load_data, load_lm, load_resume_state, save_resume)
from config import SEED
from models import load_tokenizer
from train_reward import load_reward_model

# ---- settings (Task 4: use exactly these) ----
CLIP_EPS = 0.2
BETA_KL = 0.05
C_V = 0.1
C_H = 0.01
GAMMA = 1.0
K_EPOCHS = 4
LR = 1e-5
MAX_GRAD_NORM = 1.0
N_PROMPTS = 16
N_ITERS = 200
MAX_NEW = 48

# ---- run options (do not change the algorithm) ----
N_CHUNKS = 1        # set to 2 if the GPU runs out of memory: gradients are summed before the step
SAVE_EVERY = 20     # iterations between resume saves
SAMPLE_EVERY = 20   # print one sampled summary every N iterations

SFT_CKPT = os.path.join(CHECKPOINTS_DIR, "sft")
REWARD_CKPT = os.path.join(CHECKPOINTS_DIR, "reward")
OUT = os.path.join(CHECKPOINTS_DIR, "ppo")
RESUME = os.path.join(CHECKPOINTS_DIR, "ppo_resume")
VALUE_HEAD = "value_head.pt"

LOG_FIELDS = ["iteration", "mean_reward", "mean_kl", "mean_length", "clip_frac",
              "policy_loss", "value_loss", "entropy", "first_epoch_clip_frac",
              "first_epoch_max_ratio_dev"]


class PolicyWithValue(nn.Module):
    """pi_theta (the SFT language model) + V_theta (a linear head on its last hidden layer)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.value_head = nn.Linear(lm.config.hidden_size, 1).to(lm.device)

    def save_pretrained(self, path):
        """LM in Hugging Face format + value head (also used by common.save_resume)."""
        self.lm.save_pretrained(path)
        torch.save(self.value_head.state_dict(), os.path.join(path, VALUE_HEAD))


def load_policy(path):
    policy = PolicyWithValue(load_lm(path))
    head = os.path.join(path, VALUE_HEAD)
    if os.path.isfile(head):   # resume folder; a fresh start from SFT keeps the new head
        policy.value_head.load_state_dict(torch.load(head, map_location=policy.lm.device,
                                                     weights_only=True))
    return policy


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


def build_batch(prompts, responses, pad_id, device):
    """Right-padded prompt+summary rows for scoring.

    responses[i] is the generated summary of prompts[i], already cut after its first EOS.
    Rows are padded to max(prompt_len) + T so that every index prompt_len + t (t < T)
    exists; positions past a summary's end are padding and masked out."""
    plens = [len(p["prompt_ids"]) for p in prompts]
    rlens = [len(r) for r in responses]
    T = max(rlens)
    L = max(plens) + T
    ids = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(prompts), L), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        seq = p["prompt_ids"] + r
        ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
    resp_mask = (torch.arange(T)[None, :] < torch.tensor(rlens)[:, None]).float()
    return {"ids": ids.to(device), "attn": attn.to(device), "T": T,
            "prompt_lens": torch.tensor(plens, device=device),
            "lengths": torch.tensor(rlens, device=device), "mask": resp_mask.to(device)}


def forward_tokens(lm, ids, attn, prompt_lens, T, value_head=None):
    """For each summary position t: log pi(a_t | s_t), entropy H(pi(. | s_t)) and V(s_t).

    All three are read from the hidden state that predicts a_t, i.e. index prompt_len + t - 1."""
    hidden = lm.model(input_ids=ids, attention_mask=attn).last_hidden_state       # (B, L, H)
    idx = prompt_lens[:, None] - 1 + torch.arange(T, device=ids.device)[None, :]  # (B, T)
    h = hidden.gather(1, idx[..., None].expand(-1, -1, hidden.size(-1)))          # (B, T, H)
    logp_all = F.log_softmax(lm.lm_head(h).float(), dim=-1)                       # (B, T, V)
    actions = ids.gather(1, idx + 1)                                              # a_t
    logp = logp_all.gather(-1, actions[..., None]).squeeze(-1)                    # (B, T)
    entropy = -(logp_all.exp() * logp_all).sum(-1)                                # (B, T)
    values = value_head(h).squeeze(-1) if value_head is not None else None        # (B, T)
    return logp, entropy, values


@torch.no_grad()
def rollout(policy, ref, reward_model, tok, prompts):
    """4.1 sample one summary per prompt, 4.2 store log-probs, values and r_phi."""
    responses = generate_ids(policy.lm, tok, prompts, MAX_NEW, batch_size=len(prompts),
                             do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
    b = build_batch(prompts, responses, tok.pad_token_id, policy.lm.device)
    policy.eval()
    b["logp_old"], _, b["values_old"] = forward_tokens(policy.lm, b["ids"], b["attn"],
                                                       b["prompt_lens"], b["T"], policy.value_head)
    b["logp_ref"], _, _ = forward_tokens(ref, b["ids"], b["attn"], b["prompt_lens"], b["T"])
    b["scores"] = reward_model(b["ids"], b["attn"])   # reads the last real token of each row
    b["responses"] = responses
    return b


def compute_advantages(b):
    """4.3 per-token rewards, 4.4 returns and normalised advantages."""
    mask = b["mask"]
    kl = (b["logp_old"] - b["logp_ref"]) * mask
    rewards = -BETA_KL * kl
    last = (b["lengths"] - 1)[:, None]                    # index T of each summary's final token
    rewards = rewards.scatter_add(1, last, b["scores"][:, None])
    # R_t = sum_k gamma^k r_{t+k}; padded positions have r = 0, so they add nothing.
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.size(1))):
        running = rewards[:, t] + GAMMA * running
        returns[:, t] = running
    returns = returns * mask
    adv = (returns - b["values_old"]) * mask
    mean = masked_mean(adv, mask)
    std = torch.sqrt(masked_mean((adv - mean) ** 2, mask))
    b["advantages"] = ((adv - mean) / (std + 1e-8)) * mask
    b["returns"] = returns
    b["kl_per_seq"] = kl.sum(1)                            # sum over the summary's tokens
    return b


def ppo_update(policy, opt, b):
    """4.5 K epochs on the same rollout; one optimiser step per epoch over all 16 summaries.

    Every expectation is a mean over the batch's generated tokens. With N_CHUNKS > 1 each
    chunk's sums are divided by the token count of the whole batch, so the summed gradients
    equal the full-batch gradient."""
    policy.train()
    n_tok = b["mask"].sum()
    chunks = torch.arange(b["ids"].size(0), device=b["ids"].device).chunk(N_CHUNKS)
    epochs = []
    for _ in range(K_EPOCHS):
        opt.zero_grad()
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clipped": 0.0, "max_dev": 0.0}
        for c in chunks:
            m = b["mask"][c]
            logp, entropy, values = forward_tokens(policy.lm, b["ids"][c], b["attn"][c],
                                                   b["prompt_lens"][c], b["T"], policy.value_head)
            ratio = torch.exp(logp - b["logp_old"][c])
            adv = b["advantages"][c]
            l_ppo = (torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv) * m).sum() / n_tok
            l_value = (((values - b["returns"][c]) ** 2) * m).sum() / n_tok
            l_entropy = (entropy * m).sum() / n_tok
            loss = -l_ppo + C_V * l_value - C_H * l_entropy
            loss.backward()

            outside = ((ratio < 1 - CLIP_EPS) | (ratio > 1 + CLIP_EPS)).float()
            stats["policy_loss"] += -l_ppo.item()
            stats["value_loss"] += l_value.item()
            stats["entropy"] += l_entropy.item()
            stats["clipped"] += (outside * m).sum().item()
            stats["max_dev"] = max(stats["max_dev"], ((ratio - 1).abs() * m).max().item())
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        opt.step()
        stats["clip_frac"] = stats.pop("clipped") / n_tok.item()
        epochs.append(stats)
    opt.zero_grad()
    return epochs


def write_log(log, path=os.path.join(LOGS_DIR, "ppo_log.csv")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(log)


def plot(log, path=os.path.join(FIGURES_DIR, "fig3_ppo.png")):
    it = [row["iteration"] for row in log]
    panels = [("mean_reward", "mean r_phi"), ("mean_kl", "mean KL to reference"),
              ("mean_length", "mean summary length (tokens)"), ("clip_frac", "clip fraction")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, label) in zip(axes.flat, panels):
        ax.plot(it, [row[key] for row in log])
        ax.set_xlabel("iteration")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    for path, script in ((SFT_CKPT, "train_sft.py"), (REWARD_CKPT, "train_reward.py")):
        if not os.path.isdir(path):
            raise SystemExit(f"Missing {path}. Run {script} first.")
    torch.manual_seed(SEED)
    tok = load_tokenizer()
    sft_train, _, _, _, _ = load_data(tok)

    resume_from = find_resume(RESUME)
    policy = load_policy(resume_from or SFT_CKPT)
    ref = freeze(load_lm(SFT_CKPT))
    reward_model = load_reward_model(REWARD_CKPT, SFT_CKPT, frozen=True)
    print(f"model dtype {policy.lm.dtype}", flush=True)
    print("reference frozen:", all(not p.requires_grad for p in ref.parameters()),
          "| reward model frozen:", all(not p.requires_grad for p in reward_model.parameters()),
          flush=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    order = torch.randperm(len(sft_train), generator=torch.Generator().manual_seed(SEED)).tolist()

    if resume_from:
        state = load_resume_state(resume_from, opt)
        start, log, elapsed_done = state["step"], state["log"], state["elapsed_min"]
        torch.set_rng_state(state["rng_cpu"])            # same samples as an uninterrupted run
        if torch.cuda.is_available() and state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        print(f"resuming from iteration {start}", flush=True)
    else:
        start, log, elapsed_done = 0, [], 0.0

    t0 = time.time()
    for it in range(start + 1, N_ITERS + 1):
        prompts = [sft_train[i] for i in order[(it - 1) * N_PROMPTS: it * N_PROMPTS]]
        b = compute_advantages(rollout(policy, ref, reward_model, tok, prompts))
        epochs = ppo_update(policy, opt, b)

        row = {
            "iteration": it,
            "mean_reward": b["scores"].mean().item(),
            "mean_kl": b["kl_per_seq"].mean().item(),
            "mean_length": b["lengths"].float().mean().item(),
            "clip_frac": sum(e["clip_frac"] for e in epochs) / len(epochs),
            "policy_loss": sum(e["policy_loss"] for e in epochs) / len(epochs),
            "value_loss": sum(e["value_loss"] for e in epochs) / len(epochs),
            "entropy": sum(e["entropy"] for e in epochs) / len(epochs),
            "first_epoch_clip_frac": epochs[0]["clip_frac"],
            "first_epoch_max_ratio_dev": epochs[0]["max_dev"],
        }
        log.append(row)
        elapsed = elapsed_done + (time.time() - t0) / 60
        print(f"it {it}/{N_ITERS}  r {row['mean_reward']:+.3f}  kl {row['mean_kl']:.3f}  "
              f"len {row['mean_length']:.1f}  clip {row['clip_frac']:.3f}  "
              f"pg {row['policy_loss']:+.4f}  vf {row['value_loss']:.4f}  ent {row['entropy']:.3f}  "
              f"{elapsed:.1f} min", flush=True)
        if it == 1:
            print(f"check: iteration-1 KL = {row['mean_kl']:.6f} (expect 0)", flush=True)
        if row["first_epoch_clip_frac"] > 0 or row["first_epoch_max_ratio_dev"] > 1e-3:
            print(f"WARNING: first-epoch ratio != 1 (max |r-1| = {row['first_epoch_max_ratio_dev']:.2e}, "
                  f"clip frac {row['first_epoch_clip_frac']:.3f})", flush=True)
        if it % SAMPLE_EVERY == 0:
            print("  sample:", repr(decode_summary(tok, b["responses"][0])), flush=True)
        write_log(log)
        if it % SAVE_EVERY == 0 and it < N_ITERS:
            save_resume(RESUME, policy, tok, opt, {
                "step": it, "log": log, "elapsed_min": elapsed, "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})

    train_minutes = elapsed_done + (time.time() - t0) / 60
    os.makedirs(OUT, exist_ok=True)
    policy.lm.save_pretrained(OUT)            # the policy only; the value head is not saved
    tok.save_pretrained(OUT)
    plot(log)

    worst = max(row["first_epoch_max_ratio_dev"] for row in log)
    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"check: max first-epoch |ratio - 1| over all iterations = {worst:.2e} (expect ~0)", flush=True)
    print(f"final mean r_phi {log[-1]['mean_reward']:.4f}  (mean of last 10: "
          f"{sum(r['mean_reward'] for r in log[-10:]) / len(log[-10:]):.4f})", flush=True)
    print(f"train minutes {train_minutes:.2f}  device {gpu_name}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
