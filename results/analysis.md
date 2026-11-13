# Analysis (Task 7)

Numbers below are filled in by `evaluate.py`; the answers are yours to write.

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

**Answer:** _TODO: what do the mistakes have in common (length, copying, detail)?_

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

**Answer:** _TODO: did the score rise, did length change, is a higher r_phi a better summary?_

## 7.3 The role of beta_KL

**Answer:** _TODO: reason with Eq. 41 about beta_KL = 0 (do not run it)._

## 7.4 Clip fraction

Mean clip fraction over iterations: 0.0160; max: 0.1119; first epoch of every iteration: 0.0000.

**Answer:** _TODO: what this says about the size of each policy update._

## 7.5 Where did DPO's reward go?

DPO implicit reward accuracy on the 1,000 validation pairs: **0.6100** vs reward model pairwise accuracy **0.6420**.

**Answer:** _TODO: Eqs. 55-58, why Z(x) cancels._

## 7.6 Cost and choice

| Stage | Models in GPU memory | Time |
|---|---|---|
| SFT | 1 (policy) | 10.8 min |
| Reward model | 1 (SFT body + linear head) | 20.6 min |
| PPO | 3: policy + value head (trained), reference, reward model (frozen) | 14.3 min |
| DPO | 2: policy (trained), reference (frozen) | 34.6 min |

GPU: NVIDIA A100-SXM4-80GB.

**Answer:** _TODO: which method would you choose for a new project, and why?_
