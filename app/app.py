"""Task 8.1 - web front end.

User pastes a post; shows SFT, PPO and DPO summaries side by side with reward scores.
Loads checkpoints/sft, checkpoints/ppo, checkpoints/dpo, checkpoints/reward.

    python app/app.py           # http://127.0.0.1:7860
    python app/app.py --share   # also prints a public gradio.live link (use this on Colab)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from common import CHECKPOINTS_DIR, decode_summary, generate_ids, load_lm  # noqa: E402
from batching import collate_sft  # noqa: E402
from config import MAX_PROMPT_TOKENS, MAX_SUMMARY_TOKENS  # noqa: E402
from data import clean_post  # noqa: E402
from models import load_tokenizer  # noqa: E402
from train_reward import load_reward_model  # noqa: E402

POLICIES = ("SFT", "PPO", "DPO")

EXAMPLE = (
    "relationships",
    "My [24F] boyfriend [26M] keeps cancelling plans last minute",
    "We've been together for two years. For the last three months he has cancelled almost every plan we make, "
    "usually an hour before, saying he's tired from work. When I bring it up he says I'm overreacting and that "
    "he'll make it up to me, but nothing changes. I don't want to nag him, but I'm starting to feel like I'm not "
    "a priority. How do I bring this up without starting another fight?",
)


class Summarizer:
    """Loads the three policies and the reward model once; summarises and scores a post."""

    def __init__(self, checkpoints=CHECKPOINTS_DIR):
        missing = [n for n in ("sft", "ppo", "dpo", "reward") if not os.path.isdir(os.path.join(checkpoints, n))]
        if missing:
            raise SystemExit(f"Missing checkpoints in {checkpoints}: {', '.join(missing)}. Train them first.")
        self.tok = load_tokenizer()
        self.policies = {name: load_lm(os.path.join(checkpoints, name.lower())).eval() for name in POLICIES}
        self.reward_model = load_reward_model(os.path.join(checkpoints, "reward"),
                                              os.path.join(checkpoints, "sft"), frozen=True)

    def prompt_ids(self, subreddit, title, post):
        """Same format as the training data: SUBREDDIT / TITLE / POST, then 'TL;DR:'.
        Long posts are cut so the prompt fits the 448-token limit the models were trained with."""
        sub = subreddit.strip().removeprefix("r/") or "AskReddit"
        head = f"SUBREDDIT: r/{sub}\nTITLE: {title.strip()}\nPOST: "
        budget = MAX_PROMPT_TOKENS - len(self.tok(clean_post(head), add_special_tokens=False)["input_ids"])
        body_ids = self.tok(post.strip(), add_special_tokens=False)["input_ids"]
        truncated = len(body_ids) > budget
        body = self.tok.decode(body_ids[:max(budget, 0)])
        ids = self.tok(clean_post(head + body), add_special_tokens=False)["input_ids"]
        return ids[:MAX_PROMPT_TOKENS], truncated

    @torch.no_grad()
    def __call__(self, subreddit, title, post):
        if not post.strip():
            return ["Paste a post first."] * len(POLICIES) + [""]
        ids, truncated = self.prompt_ids(subreddit, title, post)
        example = [{"prompt_ids": ids}]
        outputs = []
        for name in POLICIES:
            summary = generate_ids(self.policies[name], self.tok, example, MAX_SUMMARY_TOKENS, 1)[0]  # greedy
            b = collate_sft([{"prompt_ids": ids, "summary_ids": summary}], self.tok.pad_token_id)
            b = {k: v.to(self.reward_model.device) for k, v in b.items()}
            score = self.reward_model(b["input_ids"], b["attention_mask"]).item()
            outputs.append(f"{decode_summary(self.tok, summary).strip()}\n\nreward-model score: {score:+.2f}  "
                           f"·  {len(summary)} tokens")
        note = "Post was cut to fit the 448-token prompt limit used in training." if truncated else ""
        return outputs + [note]


def build_ui(summarize):
    import gradio as gr

    with gr.Blocks(title="TL;DR from human feedback") as demo:
        gr.Markdown("# TL;DR from human feedback\n"
                    "SmolLM2-135M fine-tuned with SFT, then aligned with **PPO** and **DPO**. "
                    "Paste a Reddit post to compare the three summaries; each is scored by the reward model.")
        with gr.Row():
            subreddit = gr.Textbox(label="Subreddit", value=EXAMPLE[0], scale=1)
            title = gr.Textbox(label="Title", value=EXAMPLE[1], scale=3)
        post = gr.Textbox(label="Post", value=EXAMPLE[2], lines=8)
        button = gr.Button("Summarize", variant="primary")
        with gr.Row():
            outs = [gr.Textbox(label=name, lines=6) for name in POLICIES]
        note = gr.Markdown()
        button.click(summarize, inputs=[subreddit, title, post], outputs=outs + [note])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link (Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui(Summarizer()).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
