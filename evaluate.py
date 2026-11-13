"""Task 6 - compare SFT, PPO and DPO on the 300 evaluation posts.

Greedy summaries (max 48 tokens) per policy; metrics: mean reward, win rate vs SFT,
KL to SFT, length, ROUGE-L. Save to results/eval_summaries.jsonl.

    python evaluate.py               # Task 6 + report files (run after all four stages)
    python evaluate.py --tally-human # Table 4, after both members filled results/human_eval.csv

Writes:
  results/eval_summaries.jsonl     every post with all three summaries and their metrics
  results/table3_final.md          Table 3
  results/table2_training.md       Table 2 (from results/logs/*_summary.json)
  results/correctness_checks.md    Section 3.3 checks, to paste into the README
  results/human_eval.csv           blind sheet for Section 3.2 (+ human_eval_key.csv, do not open
                                   before picking); never overwritten once it exists
  results/samples.md               Section 4.6 skeleton: 10 posts, "why" left for you; never overwritten
  results/analysis.md              Task 7 skeleton with the numbers filled in; never overwritten
"""
import argparse
import csv
import json
import os
import random
import sys

import torch
from rouge_score import rouge_scorer

from common import (CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR, decode_summary, generate_ids,
                    load_data, load_lm, token_logprobs)
from batching import collate_sft
from config import SEED
from models import load_tokenizer
from train_reward import load_reward_model

MAX_NEW = 48
BATCH = 16
POLICIES = ["sft", "ppo", "dpo"]
N_HUMAN = 30
N_SAMPLES = 10
HUMAN_SHEET = "human_eval.csv"
HUMAN_KEY = "human_eval_key.csv"


def ckpt(name):
    return os.path.join(CHECKPOINTS_DIR, name)


def results(name):
    return os.path.join(RESULTS_DIR, name)


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def score(policy, ref, reward_model, posts, summaries, pad_id):
    """r_phi(x, y) and KL to SFT = sum_t [log pi(y_t|x,y<t) - log pi_ref(y_t|x,y<t)] per post."""
    rewards, kls = [], []
    for i in range(0, len(posts), BATCH):
        batch = [{"prompt_ids": p["prompt_ids"], "summary_ids": s}
                 for p, s in zip(posts[i:i + BATCH], summaries[i:i + BATCH])]
        b = {k: v.to(reward_model.device) for k, v in collate_sft(batch, pad_id).items()}
        rewards += reward_model(b["input_ids"], b["attention_mask"]).tolist()
        if policy is ref:
            kls += [0.0] * len(batch)
            continue
        lp, mask = token_logprobs(policy, b["input_ids"], b["attention_mask"], b["prompt_lens"])
        lp_ref, _ = token_logprobs(ref, b["input_ids"], b["attention_mask"], b["prompt_lens"])
        kls += ((lp - lp_ref) * mask).sum(1).tolist()
    return rewards, kls


def evaluate_policies(tok, eval_posts):
    for name in POLICIES + ["reward"]:
        if not os.path.isdir(ckpt(name)):
            raise SystemExit(f"Missing {ckpt(name)}. Train every stage before evaluate.py.")
    ref = load_lm(ckpt("sft")).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    reward_model = load_reward_model(ckpt("reward"), ckpt("sft"), frozen=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    per_policy = {}
    for name in POLICIES:
        policy = ref if name == "sft" else load_lm(ckpt(name)).eval()
        ids = generate_ids(policy, tok, eval_posts, MAX_NEW, BATCH)          # greedy
        rewards, kls = score(policy, ref, reward_model, eval_posts, ids, tok.pad_token_id)
        texts = [decode_summary(tok, s) for s in ids]
        per_policy[name] = {
            "summary": texts, "reward": rewards, "kl": kls, "length": [len(s) for s in ids],
            "rouge_l": [scorer.score(p["summary"].strip(), t.strip())["rougeL"].fmeasure
                        for p, t in zip(eval_posts, texts)],
        }
        print(f"{name}: mean r {sum(rewards) / len(rewards):+.4f}", flush=True)
        if policy is not ref:
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return per_policy


def metrics_table(per_policy):
    n = len(per_policy["sft"]["reward"])
    mean = lambda xs: sum(xs) / len(xs)
    rows = {}
    for name in POLICIES:
        d = per_policy[name]
        win = None if name == "sft" else mean([float(r > s) for r, s in zip(d["reward"], per_policy["sft"]["reward"])])
        rows[name] = {"mean_reward": mean(d["reward"]), "win_rate_vs_sft": win, "kl_to_sft": mean(d["kl"]),
                      "length": mean(d["length"]), "rouge_l": mean(d["rouge_l"])}
    lines = [f"# Table 3 - Final comparison ({n} evaluation posts)", "",
             "| Policy | Mean r_phi | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |",
             "|---|---|---|---|---|---|"]
    for name in POLICIES:
        r = rows[name]
        win = "–" if r["win_rate_vs_sft"] is None else f"{r['win_rate_vs_sft']:.3f}"
        kl = "0" if name == "sft" else f"{r['kl_to_sft']:.3f}"
        lines.append(f"| {name.upper()} | {r['mean_reward']:.3f} | {win} | {kl} | {r['length']:.1f} | {r['rouge_l']:.3f} |")
    return rows, "\n".join(lines) + "\n"


# ---------------------------------------------------------------- report files

def read_summary(stage):
    path = os.path.join(LOGS_DIR, f"{stage}_summary.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, spec=".4f"):
    return "n/a" if x is None else format(x, spec)


def table2():
    rows = [("SFT", "sft", "val. loss", "val_loss"), ("Reward model", "reward", "pairwise acc.", "val_pairwise_acc"),
            ("PPO", "ppo", "final mean r_phi", "final_mean_reward"), ("DPO", "dpo", "implicit reward acc.", "val_implicit_acc")]
    lines = ["# Table 2 - Training", "", "| Stage | Final train loss | Validation metric | Time / GPU |", "|---|---|---|---|"]
    for label, stage, metric, key in rows:
        s = read_summary(stage)
        if s is None:
            lines.append(f"| {label} | not run | {metric}: not run | |")
            continue
        extra = f" (last-10 mean {s['last10_mean_reward']:.4f})" if stage == "ppo" else ""
        lines.append(f"| {label} | {fmt(s['final_train_loss'])} | {metric}: {fmt(s[key])}{extra} | "
                     f"{s['minutes']:.1f} min / {s['gpu']} |")
    return "\n".join(lines) + "\n"


def correctness_checks():
    data, ppo, dpo = read_summary("data"), read_summary("ppo"), read_summary("dpo")
    ok = lambda b: "PASS" if b else "FAIL"
    lines = ["# Correctness checks (Section 3.3)", ""]
    if data:
        lines.append(f"- **SFT mask** ({ok(data['sft_mask_check_passed'])}): decoding the tokens where "
                     f"summary_mask = 1 gives exactly the summary + EOS, e.g. `{data['mask_check_examples'][0]!r}`.")
    if ppo and dpo:
        frozen = ppo["check_reference_frozen"] and ppo["check_reward_model_frozen"] and dpo["check_reference_frozen"]
        lines.append(f"- **Frozen models** ({ok(frozen)}): every parameter of pi_ref and r_phi has "
                     f"requires_grad = False in Stage 3 (ref {ppo['check_reference_frozen']}, reward "
                     f"{ppo['check_reward_model_frozen']}) and of pi_ref in Stage 4 ({dpo['check_reference_frozen']}).")
    if ppo:
        lines.append(f"- **PPO, first iteration** ({ok(abs(ppo['check_iteration1_kl']) < 1e-6)}): "
                     f"mean KL to the reference = {ppo['check_iteration1_kl']:.2e}.")
        lines.append(f"- **PPO, every iteration** ({ok(ppo['check_max_first_epoch_ratio_dev'] < 1e-3 and ppo['check_max_first_epoch_clip_frac'] == 0)}): "
                     f"in the first of the K epochs, max |r_t(theta) - 1| over all iterations = "
                     f"{ppo['check_max_first_epoch_ratio_dev']:.2e}, max clip fraction = {ppo['check_max_first_epoch_clip_frac']}.")
    if dpo and dpo["check_first_step_loss"] is not None:
        lines.append(f"- **DPO, first step** ({ok(abs(dpo['check_first_step_loss'] - dpo['expected_first_step_loss']) < 1e-3)}): "
                     f"loss = {dpo['check_first_step_loss']:.4f} (log 2 = {dpo['expected_first_step_loss']:.4f}).")
    lines.append("- **Padding**: prompts for `generate` are left-padded (`collate_prompts`); the reward model reads the "
                 "hidden state at index `attention_mask.sum(1) - 1` of right-padded rows, i.e. the last real token. "
                 "Real tokens are always found with `attention_mask`, never with `input_ids != pad_token_id`.")
    return "\n".join(lines) + "\n"


def write_human_sheet(eval_posts, per_policy):
    sheet, key = results(HUMAN_SHEET), results(HUMAN_KEY)
    if os.path.exists(sheet):
        print(f"kept existing {sheet}", flush=True)
        return
    rng = random.Random(SEED)
    items = rng.sample(range(len(eval_posts)), N_HUMAN)
    with open(sheet, "w", newline="", encoding="utf-8") as fs, open(key, "w", newline="", encoding="utf-8") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["item", "post", "summary_A", "summary_B", "summary_C", "member1_pick", "member2_pick"])
        wk.writerow(["item", "eval_index", "A", "B", "C"])
        for n, idx in enumerate(items, start=1):
            order = POLICIES[:]
            rng.shuffle(order)
            ws.writerow([n, eval_posts[idx]["prompt"]] + [per_policy[m]["summary"][idx].strip() for m in order] + ["", ""])
            wk.writerow([n, idx] + order)


def write_samples(eval_posts, per_policy):
    path = results("samples.md")
    if os.path.exists(path):
        print(f"kept existing {path}", flush=True)
        return
    sft = per_policy["sft"]["reward"]
    gain = [max(per_policy["ppo"]["reward"][i], per_policy["dpo"]["reward"][i]) - sft[i] for i in range(len(sft))]
    loss = [min(per_policy["ppo"]["reward"][i], per_policy["dpo"]["reward"][i]) - sft[i] for i in range(len(sft))]
    better = sorted(range(len(sft)), key=lambda i: -gain[i])[:3]
    worse = [i for i in sorted(range(len(sft)), key=lambda i: loss[i]) if i not in better][:3]
    rest = [i for i in random.Random(SEED + 1).sample(range(len(sft)), len(sft)) if i not in better + worse]
    picks = [(i, "reward model prefers PPO/DPO most over SFT") for i in better] + \
            [(i, "reward model prefers SFT most over PPO/DPO") for i in worse] + \
            [(i, "random") for i in rest[:N_SAMPLES - len(better) - len(worse)]]
    out = ["# Qualitative samples (Section 4.6)", "",
           "Posts were picked by reward-model score difference (a proxy). Read each one and say in your own words "
           "whether PPO or DPO is *actually* better or worse than SFT and why. Keep at least two clearly better "
           "and two clearly worse examples; swap posts in from `eval_summaries.jsonl` if needed.", ""]
    for n, (i, why) in enumerate(picks, start=1):
        out += [f"## Sample {n} (eval post {i}; picked because: {why})", "", "```", eval_posts[i]["prompt"], "```", "",
                f"- **Human reference:** {eval_posts[i]['summary'].strip()}"]
        for m in POLICIES:
            d = per_policy[m]
            out.append(f"- **{m.upper()}** (r_phi {d['reward'][i]:+.2f}, {d['length'][i]} tokens): {d['summary'][i].strip()}")
        out += ["", "**Why:** _TODO_", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def write_analysis_skeleton(rows):
    path = results("analysis.md")
    if os.path.exists(path):
        print(f"kept existing {path}", flush=True)
        return
    rew, ppo, dpo, sft = (read_summary(s) or {} for s in ("reward", "ppo", "dpo", "sft"))
    out = ["# Analysis (Task 7)", "",
           "Numbers below are filled in by `evaluate.py`; the answers are yours to write.", ""]

    out += ["## 7.1 Reward-model accuracy and its mistakes", "",
            f"Pairwise accuracy on 1,000 validation pairs: **{fmt(rew.get('val_pairwise_acc'))}** (chance = 0.5).", "",
            "Three validation pairs it gets most confidently wrong (from `results/logs/reward_val_scores.jsonl`):", ""]
    scores_path = os.path.join(LOGS_DIR, "reward_val_scores.jsonl")
    if os.path.isfile(scores_path):
        with open(scores_path, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        for n, p in enumerate(sorted(pairs, key=lambda p: p["r_chosen"] - p["r_rejected"])[:3], start=1):
            out += [f"{n}. r(chosen) = {p['r_chosen']:+.2f}, r(rejected) = {p['r_rejected']:+.2f}",
                    f"   - post: {p['prompt'][-300:]!r}",
                    f"   - chosen (human preferred): {p['chosen'].strip()!r}",
                    f"   - rejected: {p['rejected'].strip()!r}"]
    out += ["", "**Answer:** _TODO: what do the mistakes have in common (length, copying, detail)?_", ""]

    log_path = os.path.join(LOGS_DIR, "ppo_log.csv")
    ppo_rows = []
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8") as f:
            ppo_rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    avg = lambda rs, k: sum(r[k] for r in rs) / len(rs) if rs else None
    first, last = ppo_rows[:10], ppo_rows[-10:]
    out += ["## 7.2 PPO: reward, KL and length", "", "See `results/figures/fig3_ppo.png`.", "",
            "| | first 10 iterations | last 10 iterations |", "|---|---|---|",
            f"| mean r_phi | {fmt(avg(first, 'mean_reward'))} | {fmt(avg(last, 'mean_reward'))} |",
            f"| mean KL | {fmt(avg(first, 'mean_kl'))} | {fmt(avg(last, 'mean_kl'))} |",
            f"| mean length | {fmt(avg(first, 'mean_length'), '.1f')} | {fmt(avg(last, 'mean_length'), '.1f')} |", "",
            f"Evaluation (Table 3): mean r_phi SFT {rows['sft']['mean_reward']:.3f} vs PPO {rows['ppo']['mean_reward']:.3f}; "
            f"ROUGE-L SFT {rows['sft']['rouge_l']:.3f} vs PPO {rows['ppo']['rouge_l']:.3f}; "
            f"length SFT {rows['sft']['length']:.1f} vs PPO {rows['ppo']['length']:.1f}. "
            "Human evaluation: `results/table4_human_eval.md`.", "",
            "**Answer:** _TODO: did the score rise, did length change, is a higher r_phi a better summary?_", "",
            "## 7.3 The role of beta_KL", "", "**Answer:** _TODO: reason with Eq. 41 about beta_KL = 0 (do not run it)._", "",
            "## 7.4 Clip fraction", "",
            f"Mean clip fraction over iterations: {fmt(avg(ppo_rows, 'clip_frac'))}; "
            f"max: {fmt(max((r['clip_frac'] for r in ppo_rows), default=None))}; "
            f"first epoch of every iteration: {fmt(ppo.get('check_max_first_epoch_clip_frac'))}.", "",
            "**Answer:** _TODO: what this says about the size of each policy update._", "",
            "## 7.5 Where did DPO's reward go?", "",
            f"DPO implicit reward accuracy on the 1,000 validation pairs: **{fmt(dpo.get('val_implicit_acc'))}** vs "
            f"reward model pairwise accuracy **{fmt(rew.get('val_pairwise_acc'))}**.", "",
            "**Answer:** _TODO: Eqs. 55-58, why Z(x) cancels._", "",
            "## 7.6 Cost and choice", "",
            "| Stage | Models in GPU memory | Time |", "|---|---|---|",
            f"| SFT | 1 (policy) | {fmt(sft.get('minutes'), '.1f')} min |",
            f"| Reward model | 1 (SFT body + linear head) | {fmt(rew.get('minutes'), '.1f')} min |",
            f"| PPO | 3: policy + value head (trained), reference, reward model (frozen) | {fmt(ppo.get('minutes'), '.1f')} min |",
            f"| DPO | 2: policy (trained), reference (frozen) | {fmt(dpo.get('minutes'), '.1f')} min |", "",
            f"GPU: {sft.get('gpu', 'n/a')}.", "",
            "**Answer:** _TODO: which method would you choose for a new project, and why?_", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def tally_human():
    sheet, key = results(HUMAN_SHEET), results(HUMAN_KEY)
    with open(key, encoding="utf-8") as f:
        key_rows = {r["item"]: r for r in csv.DictReader(f)}
    with open(sheet, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = {m: {p: 0 for p in POLICIES} for m in ("member1_pick", "member2_pick")}
    agree = both = 0
    for r in rows:
        picks = {}
        for m in counts:
            letter = r[m].strip().upper()
            if letter in ("A", "B", "C"):
                picks[m] = key_rows[r["item"]][letter]
                counts[m][picks[m]] += 1
        if len(picks) == 2:
            both += 1
            agree += picks["member1_pick"] == picks["member2_pick"]
    lines = [f"# Table 4 - Human evaluation ({len(rows)} posts)", "",
             "| | SFT picked | PPO picked | DPO picked |", "|---|---|---|---|"]
    for n, m in enumerate(counts, start=1):
        lines.append(f"| Member {n} | " + " | ".join(str(counts[m][p]) for p in POLICIES) + " |")
    lines.append(f"| Agreement between members | {agree} / {both} | | |")
    text = "\n".join(lines) + "\n"
    with open(results("table4_human_eval.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tally-human", action="store_true", help="build Table 4 from the filled human_eval.csv")
    args = parser.parse_args()
    if args.tally_human:
        tally_human()
        return

    torch.manual_seed(SEED)
    tok = load_tokenizer()
    eval_posts = load_data(tok)[4]
    per_policy = evaluate_policies(tok, eval_posts)
    rows, table = metrics_table(per_policy)

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(results("eval_summaries.jsonl"), "w", encoding="utf-8") as f:
        for i, post in enumerate(eval_posts):
            rec = {"index": i, "prompt": post["prompt"], "reference": post["summary"].strip()}
            for m in POLICIES:
                d = per_policy[m]
                rec[m] = {"summary": d["summary"][i].strip(), "reward": d["reward"][i], "kl_to_sft": d["kl"][i],
                          "length": d["length"][i], "rouge_l": d["rouge_l"][i]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(LOGS_DIR, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    for name, text in (("table3_final.md", table), ("table2_training.md", table2()),
                       ("correctness_checks.md", correctness_checks())):
        with open(results(name), "w", encoding="utf-8") as f:
            f.write(text)
    write_human_sheet(eval_posts, per_policy)
    write_samples(eval_posts, per_policy)
    write_analysis_skeleton(rows)
    print(table)
    print(table2())
    print(correctness_checks())


if __name__ == "__main__":
    sys.exit(main())
