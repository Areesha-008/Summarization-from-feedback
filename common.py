"""Task 1.3 - shared helpers used by every stage.

Run `python common.py` to produce Table 1 (results/table1_data.md) and the
SFT-mask correctness check.
"""
import json
import os
import pickle
import shutil
import sys

import torch

# ---- Project paths (absolute, so scripts work from any working directory) ----
ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")   # never committed
DATA_CACHE = os.path.join(ROOT, "data_cache", "data.pkl")
RESULTS_DIR = os.path.join(ROOT, "results")           # tables, summaries, analysis
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")          # per-stage training logs (csv)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")    # plots for the report / blog

sys.path.insert(0, os.path.join(ROOT, "starter"))
from batching import collate_prompts  # noqa: E402
from data import load_sft_splits, load_preference_splits, load_eval_posts  # noqa: E402
from models import load_causal_lm  # noqa: E402


def load_lm(name_or_path=None):
    """load_causal_lm, always in float32.

    SmolLM2 is stored in bfloat16 and transformers >= 5 loads checkpoints in their
    stored dtype by default. Training in bfloat16 rounds most small AdamW updates
    away, and the T4 has no native bfloat16, so every stage casts to float32."""
    model = load_causal_lm(name_or_path) if name_or_path else load_causal_lm()
    return model.float()


def load_data(tok, cache=DATA_CACHE):
    """Return (sft_train, sft_val, pref_train, pref_val, eval_posts).
    Tokenising takes minutes, so the result is pickled on the first call."""
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    sft_train, sft_val = load_sft_splits(tok)
    pref_train, pref_val = load_preference_splits(tok)
    eval_posts = load_eval_posts(tok)
    data = (sft_train, sft_val, pref_train, pref_val, eval_posts)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


# ---- Resume checkpoints (Colab / Kaggle runtimes can disconnect mid-training) ----
# A resume folder holds the trainable model (save_pretrained), the tokenizer and
# trainer.pt = optimizer state + whatever loop state the script passes in.
# Saves go to <dir>_tmp first and are swapped in, so a disconnect during a save
# still leaves the previous complete copy.

def _resume_step(path):
    meta = os.path.join(path, "step.json")
    if os.path.isfile(meta):
        with open(meta) as f:
            return json.load(f)["step"]
    pt = os.path.join(path, "trainer.pt")  # folders saved before step.json existed
    if os.path.isfile(pt):
        return torch.load(pt, map_location="cpu", weights_only=True)["step"]
    return None


def find_resume(resume_dir):
    """Return the copy of resume_dir with the highest finished step, or None."""
    best, best_step = None, -1
    for path in (resume_dir, resume_dir + "_tmp", resume_dir + "_bak"):
        if not os.path.isfile(os.path.join(path, "trainer.pt")):
            continue
        step = _resume_step(path)
        if step is not None and step > best_step:
            best, best_step = path, step
    return best


def save_resume(resume_dir, model, tok, opt, state):
    """Save model + tokenizer + optimizer + loop `state` (a dict that must contain 'step')."""
    tmp, bak = resume_dir + "_tmp", resume_dir + "_bak"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    torch.save({**state, "optimizer": opt.state_dict()}, os.path.join(tmp, "trainer.pt"))
    with open(os.path.join(tmp, "step.json"), "w") as f:  # written last: marks the copy complete
        json.dump({"step": state["step"]}, f)
    if os.path.isdir(bak):
        shutil.rmtree(bak)
    if os.path.isdir(resume_dir):
        shutil.move(resume_dir, bak)
    shutil.move(tmp, resume_dir)
    if os.path.isdir(bak):
        shutil.rmtree(bak)
    print(f"saved resume at step {state['step']}", flush=True)


def load_resume_state(path, opt):
    """Load the optimizer state from a resume folder into `opt` and return the loop state.
    Load the model itself with load_lm(path) before creating `opt`."""
    state = torch.load(os.path.join(path, "trainer.pt"), map_location="cpu", weights_only=True)
    opt.load_state_dict(state.pop("optimizer"))
    device = next(p.device for group in opt.param_groups for p in group["params"])
    for s in opt.state.values():
        for key, value in s.items():
            if torch.is_tensor(value) and key != "step":
                s[key] = value.to(device)
    return state


@torch.no_grad()
def generate_ids(model, tok, examples, max_new_tokens=48, batch_size=16, **gen_kwargs):
    """Generate a continuation of each example's prompt_ids (left-padded batches).
    Returns one token-id list per example, cut after its first EOS (EOS kept).
    Greedy by default; pass do_sample=True, temperature=..., etc. for sampling."""
    gen_kwargs.setdefault("do_sample", False)
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(examples), batch_size):
        b = collate_prompts(examples[i:i + batch_size], tok.pad_token_id)
        ids, attn = b["input_ids"].to(model.device), b["attention_mask"].to(model.device)
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **gen_kwargs)
        for row in gen[:, ids.shape[1]:].tolist():
            if tok.eos_token_id in row:
                row = row[:row.index(tok.eos_token_id) + 1]
            out.append(row)
    model.train(was_training)
    return out


def decode_summary(tok, ids):
    """Summary text without the trailing EOS."""
    if ids and ids[-1] == tok.eos_token_id:
        ids = ids[:-1]
    return tok.decode(ids)


def generate_summaries(model, tok, examples, max_new_tokens=48, batch_size=16, **gen_kwargs):
    """Like generate_ids, but returns decoded summary strings."""
    return [decode_summary(tok, ids)
            for ids in generate_ids(model, tok, examples, max_new_tokens, batch_size, **gen_kwargs)]


def write_stage_summary(stage, fields):
    """results/logs/<stage>_summary.json: final numbers + correctness checks of one stage.
    evaluate.py turns these into Table 2 and results/correctness_checks.md."""
    fields = dict(fields, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, f"{stage}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2)


def token_logprobs(model, input_ids, attention_mask, prompt_lens):
    """Return log pi(x_t | x_<t) for every position t >= 1, shape (B, L-1),
    and a mask of the same shape that is 1 exactly on summary tokens
    (not prompt, not padding).

    Position t of the outputs refers to token input_ids[:, t+1], which is
    predicted by the logits at input position t."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    logprobs = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    pos = torch.arange(1, input_ids.size(1), device=input_ids.device)  # index of the target token
    is_summary = pos[None, :] >= prompt_lens.to(input_ids.device)[:, None]
    summary_mask = (is_summary & (attention_mask[:, 1:] == 1)).long()  # pad == eos, so use the mask
    return logprobs, summary_mask


if __name__ == "__main__":
    from models import load_tokenizer
    from batching import collate_sft

    tok = load_tokenizer()
    sft_train, sft_val, pref_train, pref_val, eval_posts = load_data(tok)

    # ---- Task 1.2: Table 1 ----
    p_lens = [len(ex["prompt_ids"]) for ex in sft_train]
    s_lens = [len(ex["summary_ids"]) for ex in sft_train]
    table = "\n".join([
        "| | |", "|---|---|",
        f"| SFT train / validation pairs | {len(sft_train)} / {len(sft_val)} |",
        f"| Preference train / validation pairs | {len(pref_train)} / {len(pref_val)} |",
        f"| Evaluation posts | {len(eval_posts)} |",
        f"| Mean / max prompt length, SFT train (tokens) | {sum(p_lens) / len(p_lens):.1f} / {max(p_lens)} |",
        f"| Mean / max summary length, SFT train (tokens) | {sum(s_lens) / len(s_lens):.1f} / {max(s_lens)} |",
    ])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "table1_data.md"), "w", encoding="utf-8") as f:
        f.write("# Table 1 - Data\n\n" + table + "\n")
    print(table)

    # ---- Task 1.3: SFT mask check ----
    model = load_lm()
    model.eval()
    batch = sft_train[:8]
    b = collate_sft(batch, tok.pad_token_id)
    b = {k: v.to(model.device) for k, v in b.items()}
    with torch.no_grad():
        logprobs, mask = token_logprobs(model, b["input_ids"], b["attention_mask"], b["prompt_lens"])
    print("\nlogprobs", tuple(logprobs.shape), "mask", tuple(mask.shape))

    targets = b["input_ids"][:, 1:]
    all_ok = True
    for i, ex in enumerate(batch):
        decoded = tok.decode(targets[i][mask[i].bool()])
        ok = decoded == ex["summary"] + tok.eos_token and mask[i].sum().item() == len(ex["summary_ids"])
        all_ok &= ok
        print(f"[{i}] {'OK ' if ok else 'BAD'} n_masked={mask[i].sum().item()} {decoded!r}")
    print("\nSFT mask check:", "PASSED" if all_ok else "FAILED")
    write_stage_summary("data", {"sft_mask_check_passed": bool(all_ok),
                                 "mask_check_examples": [tok.decode(targets[i][mask[i].bool()]) for i in range(2)]})
