Can a 135M-parameter model learn what a *good* summary is?

For our Generative AI course we rebuilt the RLHF pipeline from "Learning to Summarize from Human Feedback" (the recipe behind ChatGPT) on SmolLM2-135M — writing SFT, the reward model, PPO and DPO ourselves in PyTorch, no RLHF libraries.

What we found on 300 unseen Reddit posts:
🔹 PPO and DPO beat the SFT model on the reward model ~90% of the time
🔹 DPO got 91% of PPO's gain with two models instead of three and a smaller drift from SFT
🔹 Our reward model (64% accurate) had learned that "longer is better" — and PPO happily exploited it: longer summaries, more truncation, slightly lower ROUGE
🔹 The KL penalty is what kept PPO from running off: the KL rose, then levelled off instead of exploding

Biggest debugging lesson: our first DPO run sat at log 2 (≈0.693) for 1,000 steps. transformers 5 loads SmolLM2 in bfloat16, where tiny updates round to zero. float32 fixed it.

📝 Blog: <MEDIUM LINK>
💻 Code + Gradio demo: https://github.com/Areesha-008/Summarization-from-feedback

#RLHF #LLM #PPO #DPO #MachineLearning #NLP #PyTorch
