# Analysis (Task 7)

Numbers come from `evaluate.py` and the training logs (one A100 run).

## 7.1 Reward-model accuracy and its mistakes

Pairwise accuracy on 1,000 validation pairs: **0.6420** (chance = 0.5).

Three validation pairs it gets most confidently wrong (from `results/logs/reward_val_scores.jsonl`):

1. r(chosen) = -1.50, r(rejected) = +1.90
   - post: "uesday).\n\nI don't know where I can go for support or who to talk to. I have a few casual friends but no one who would let me vent like I feel I need. No one to come with me. Do i look internally for the support? \n\nAny advice would be great. Thanks reddit. (yep, it's a throwaway, no gw posts).\nTL;DR:"
   - chosen (human preferred): "I don't have good friends and I'm losing my mind. I've been depressed for most of my life and now I feel like shit and have no idea what to do."
   - rejected: "[30F] Cut family out of life and have strained relationship with siblings. I don't know where to go for support or who to talk to."
2. r(chosen) = -0.51, r(rejected) = +2.55
   - post: 'Y church!" or "No, they\'ll all be on drugs too!"\n\nI feel like this is really the wrong thing to do in response, but I don\'t know what to do otherwise. If I don\'t do that, then I just sit on the phone with her for an hour and tell her that I\'m here for her. I don\'t feel like that helps either.\nTL;DR:'
   - chosen (human preferred): 'How can I be there for my girlfriend in hard times without physically being there?'
   - rejected: "Girlfriend has severe depression and anxiety. I try to be there for her, but she gets panic attacks from seeing the world of drugs and alcohol and leaves me. I don't know what I can do to be more supportive."
3. r(chosen) = -0.60, r(rejected) = +2.23
   - post: 'with my friends) so I started to chuckle and said "Bullshit!" and pointed at the cop asking me things. He asked me to step outside in my undies and gave me a field sobriety test and then questioned me some more. Couldn\'t sleep the rest of the night and then had an early class and a fun story.\nTL;DR:'
   - chosen (human preferred): 'TIFU by thinking I was being pranked.'
   - rejected: 'drunk asshole broke into my house, knocked over my kitchen table and left a trail of skittles through my yard. Thought it was a prank. Told the cops I was drunk.'

Supporting numbers (computed from `reward_val_scores.jsonl`):

- When the human-preferred summary is the **longer** one (579 pairs) the reward model is right 76.5% of the time; when it is the **shorter** one (373 pairs) only 46.4%.
- Over pairs whose summaries differ in length, the reward model scores the longer summary higher in 67.5% of them.

**Answer:** The reward model is clearly above chance (0.642 vs 0.5) but far from the large models in the paper, as expected
for 135M parameters and 20,000 pairs. Its three most confident mistakes share one pattern: the summary the human
preferred is **short and abstract** ("TIFU by thinking I was being pranked", "How can I be there for my girlfriend…"),
while the rejected one is **longer and packed with concrete details copied from the post** ("[30F] Cut family out of
life…", "drunk asshole broke into my house… trail of skittles…"). The model has learned "more specific, more words =
better", which is often true in the training data but not always. The numbers above confirm it is a length bias: it
is right 76.5% of the time when the preferred summary is the longer one and only 46.4% (below chance) when it is the
shorter one, and overall it scores the longer summary higher in 67.5% of pairs.

## 7.2 PPO: reward, KL and length

See `results/figures/fig3_ppo.png`.

| | first 10 iterations | last 10 iterations |
|---|---|---|
| mean r_phi | 0.7133 | 1.9884 |
| mean KL | 0.8398 | 16.9619 |
| mean length | 35.2 | 41.7 |

Evaluation (Table 3): mean r_phi SFT 0.571 vs PPO 1.932; ROUGE-L SFT 0.256 vs PPO 0.242; length SFT 30.6 vs PPO 38.9. Human evaluation: `results/table4_human_eval.md`.

Supporting numbers:

- Rollout KL passes 10 nats per summary at iteration 34 and then levels off around 15-20 (max 22.2); it does not keep growing. Greedy KL to SFT on the evaluation posts is 5.16 (PPO) and 3.19 (DPO).
- Summaries that hit the 48-token limit (cut off mid-sentence): SFT 15, PPO 77, DPO 68 of 300.
- Mean words per summary: human reference 26.6, SFT 23.9, PPO 31.5, DPO 31.6.
- Correlation between r_phi and summary length over all 900 evaluation summaries: 0.27.

**Answer:** **The score rose.** Mean r_phi went from 0.71 (first 10 iterations) to about 2.0, almost all of it in the first
~60 iterations, after which it plateaued. On the 300 evaluation posts PPO raises r_phi from 0.57 to 1.93 and beats
SFT on 90% of posts. At the same time the rollout KL jumped to ~15–20 nats per summary around iteration 55 and then
stopped growing: the KL penalty (β·KL ≈ 0.05 × 17 ≈ 0.85 per summary) balances the reward gain.

**Length changed.** Summaries grew from 30.6 to 38.9 tokens (31.5 vs 23.9 words; the human references average 26.6
words), and 77 of 300 PPO summaries hit the 48-token limit and are cut off mid-sentence, against 15 for SFT. r_phi
and length are positively correlated (0.27). Given the reward model's length bias (7.1), part of PPO's gain is the
policy exploiting that bias.

**A higher r_phi is not the same as a better summary.** ROUGE-L against the human reference falls slightly (0.256 →
0.242), and the qualitative samples show both sides: PPO is often more specific and more faithful than SFT (samples
1, 2, 3, 9), but its failures — repeating a phrase, drifting into minor details, being truncated (samples 4, 5) — are
exactly the things the reward model does not penalise. Human evaluation (Table 4): _TODO after the blind evaluation —
how often each member picked PPO vs SFT, and whether that agrees with the 90% reward-model win rate._

## 7.3 The role of beta_KL

**Answer:** Eq. 41 maximises E[r_phi(x, y)] − β_KL · KL(π_θ ‖ π_ref). In our implementation this is the per-token penalty
−β_KL(log π_θold − log π_ref) added to every generated token. β_KL is a leash: every token the policy makes more
likely than SFT costs reward, so the policy only moves away from SFT where the reward model pays more than the
penalty. That keeps the text fluent (SFT is a good language model), keeps the summaries close to the distribution the
reward model was trained on (so its scores stay meaningful), and stops the policy from collapsing onto whatever the
reward model happens to over-score. The KL-regularised optimum is π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β_KL): a reweighted
SFT model, not a new one.

With **β_KL = 0** the objective is just "maximise r_phi". The optimum becomes the single highest-scoring output for
each prompt, with nothing pulling the policy back. We would expect: the reward to climb faster and higher; the KL to
keep growing without bound instead of levelling off at ~17 nats; summaries to run to the 48-token limit (the reward
model prefers longer summaries); repetition, copied text and eventually ungrammatical output that the 135M reward
model still scores highly — classic reward hacking. Only the small entropy bonus (c_H = 0.01) and the PPO clip would
slow this down, and neither limits how far the policy drifts in total.

## 7.4 Clip fraction

Mean clip fraction over iterations: 0.0160; max: 0.1119; first epoch of every iteration: 0.0000.

**Answer:** The clip fraction is the share of generated tokens whose ratio r_t(θ) = π_θ/π_θold left [0.8, 1.2]. In the first
epoch of every iteration it is exactly 0 (the policy has not changed yet, so r_t = 1). Averaged over the four epochs
it starts at 0.11 in iteration 1 — the value head is new and the advantages are large and noisy — and then settles
at about 0.01–0.02 for the rest of training (mean 0.016). So after the first few iterations only 1–2% of tokens change
probability by more than 20% within one rollout: each update is small, the clipped objective is rarely active, and the
step size is mostly set by the learning rate (1e-5) and gradient clipping rather than by the clip. That is the regime
PPO is designed for — a trust region that acts as a safety net, not as the main brake.

## 7.5 Where did DPO's reward go?

DPO implicit reward accuracy on the 1,000 validation pairs: **0.6100** vs reward model pairwise accuracy **0.6420**.

**Answer:** The KL-constrained objective (Eq. 41) has a closed-form optimum, π*(y|x) = π_ref(y|x)·exp(r(x,y)/β) / Z(x), where
Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β) normalises over **all possible summaries** — impossible to compute. Solving for the
reward gives r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x). So every policy defines an implicit reward: **the reward
"went into" the log-ratio between the policy and the reference.** Substituting into the Bradley–Terry model, only the
difference r(x,y_w) − r(x,y_l) appears; both summaries share the same prompt x, so the β·log Z(x) terms are equal and
cancel. What remains — σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x))) — is the DPO loss, which needs
only the two models' log-probabilities of the two summaries.

DPO's implicit reward ranks the 1,000 validation pairs correctly 61.0% of the time, against 64.2% for the explicit
reward model. It reaches ~0.62 after only 250 steps and then stays flat, while the reward model keeps improving to the
end of its epoch. DPO optimises the policy, not the ranking; with β = 0.1 and lr 2e-6 its log-ratios stay small (KL to
SFT 3.2 vs 5.2 for PPO). Even so, DPO reaches nearly the same score on our reward model (1.81 vs 1.93) without ever
seeing it, because both learned from the same preference data — including the preference for longer summaries (DPO
length 39.3 tokens).

## 7.6 Cost and choice

| Stage | Models in GPU memory | Time |
|---|---|---|
| SFT | 1 (policy) | 10.8 min |
| Reward model | 1 (SFT body + linear head) | 20.6 min |
| PPO | 3: policy + value head (trained), reference, reward model (frozen) | 14.3 min |
| DPO | 2: policy (trained), reference (frozen) | 34.6 min |

GPU: NVIDIA A100-SXM4-80GB.

**Answer:** PPO needed **three models in GPU memory** — the trainable policy with its value head, the frozen reference and the
frozen reward model — plus a separate reward-model stage before it. DPO needed **two**: the trainable policy and the
frozen reference, and no reward model at all. On the A100, the PPO route took 20.6 min (reward model) + 14.3 min (PPO,
3,200 prompts) ≈ 35 min, about the same as DPO's 34.6 min (one epoch over 20,000 pairs); SFT took 10.8 min.

For a new project we would start with **DPO**. It reaches almost the same reward-model score (1.81 vs 1.93) and win rate
(89% vs 90%) with a smaller KL to SFT (3.2 vs 5.2) and slightly better ROUGE-L, and it is plain supervised training:
no sampling loop, no value head, fewer hyperparameters (one β instead of ε, β_KL, c_V, c_H, K) and no reward model to
reward-hack. PPO is worth its extra complexity when the reward is not a fixed preference dataset — when you can score
fresh samples online (a reward model you keep updating, unit tests, a human in the loop) or want to keep improving on
new prompts.
