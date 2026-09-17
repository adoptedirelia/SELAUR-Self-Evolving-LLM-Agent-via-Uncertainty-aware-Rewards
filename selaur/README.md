# The SELAUR algorithm

This package is the algorithmic core of the repository and the reference
implementation of [arXiv:2602.21158](https://arxiv.org/abs/2602.21158).
Everything else — `verl/` (the RL engine), `agent_system/` (environments and
rollout) — is infrastructure that feeds it.

## The idea in one paragraph

A multi-turn agent that succeeds is not necessarily an agent that *knew what it
was doing*. Reward alone cannot tell a deliberate solution from a lucky one, or
a near-miss from a hopeless attempt. But the policy leaves a record of its own
conviction in every rollout: the log-probabilities of the tokens it emitted.
SELAUR reads that record, turns it into a per-step uncertainty score, and folds
it into the group-relative advantage that GiGPO already computes — so the agent
is trained not only to reach the goal, but to stop guessing on the way there.

## Pipeline

```
                    ┌───────────────────────────────────────────────┐
  rollout batch ───►│  B steps, flattened; grouped by task (index)  │
                    │  and by episode (traj_index)                  │
                    └───────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
   token_level_rewards            step_rewards                rollout logprobs
   (episode outcome)         (discounted return-to-go)         (top-1, top-2)
          │                             │                             │
          │                             │                     uncertainty/signals
          │                             │                   entropy · nll · top-2
          │                             │                             │
          │                             │                  uncertainty/aggregation
          │                             │              discounted smoothing per episode
          │                             │                             │
          │                             │                  uncertainty/transforms
          │                             │              linear / neg  ·  exp
          │                             │                             │
          │                             │                   ┌─────────┴─────────┐
          ▼                             ▼                   ▼                   ▼
   episode score  ◄───────────────── (+) ───────────  episode term        step term
          │                             │                                       │
          │                        step score  ◄────────────── (+) ─────────────┘
          │                             │
  group by `index`              group by anchor state
  (normalization.py)             (grouping.py → normalization.py)
          │                             │
          └──────────► episode adv + w · step adv ──────────► token-level advantage
```

## The three parts

### 1. Two levels of grouping (inherited from GiGPO)

No critic is trained. Instead, each score is baselined against the mean of the
rows it is comparable to:

* **Episode level.** All rollouts of the same task share a prompt `uid`. The
  group mean is the task's difficulty; what survives is "was this rollout
  better than the others of the same task".
* **Step level.** Steps that observed an *identical anchor state* are grouped,
  regardless of which episode they came from (`grouping.py`). Same situation,
  different actions — so the group mean isolates the action's contribution.

Both reduce to the same operation in `normalization.py`; they differ only in
the group key and in how a singleton group is baselined.

### 2. Uncertainty (the SELAUR contribution)

`uncertainty/signals.py` reads the top-1 and top-2 log-probabilities of a step
and scores the paper's three metrics — how flat the distribution was
(**entropy**, `step_entropy`), how unlikely the chosen token was
(**least-confidence**, `step_nll`), and how close the runner-up came
(**margin**, `step_top2_margin`). `combined_uncertainty` blends them by
`w1/w2/w3` into the paper's *combined token-level uncertainty estimate*, with
`rho` sliding between "weighted average" and "whichever fired loudest".

`uncertainty/aggregation.py` then smooths the per-step scores backwards along
the episode, so a step is credited with the hesitation that followed it. The
smoothing is *normalised* by its own weight sum, which keeps the output on the
same scale for a 5-step and a 50-step episode — that is what makes
`step_weight` mean the same thing across environments.

### 3. Failure-aware reshaping (`uncertainty/transforms.py` + `advantage.py`)

The transform decides how uncertainty enters the reward, and thereby which
episodes it touches — this is the paper's *failure-aware reward reshaping*,
injected at both the step and the trajectory level:

| `transform` | applies to | effect | role |
| --- | --- | --- | --- |
| `linear` | failed episodes | additive, `+u` | the main method |
| `neg` | failed episodes | additive, `-u` | control arm: does the direction matter? |
| `exp` | successful episodes | multiplicative, `exp(-u)` | discount lucky wins |

Two details are easy to miss and both are deliberate:

* **Both levels shape in the same direction.** The episode term and the step
  term are folded in with the same sign; which direction that is, is the
  transform's decision (`linear` credits uncertainty, `neg` charges for it),
  not the level's. An ablation therefore only has to flip one knob, and the two
  levels can never quietly cancel each other out.
* **Shaping happens before normalisation**, so it lives inside the baseline.
  Shifting a whole group by the same amount changes nothing; only *relative*
  certainty within a group moves the gradient.

Finally, `threshold_success_rate` gates the whole thing: while the batch
success rate is at or below it, shaping is skipped. Early in training almost
everything fails and "how sure was it" carries no information.

Because each transform owns exactly one outcome bucket, the other bucket has to
pass through untouched — and what "untouched" means depends on the family.
Adding `0` leaves a score alone, but *multiplying* by `0` destroys it. So the
estimator fills the skipped episodes with the transform's **neutral term**
(`neutral_term()`: `0` for the additive family, `1` for the multiplicative
one). Without this, `exp` would not merely discount lucky wins — it would also
erase every failed episode's score, and with it the gradient signal from
losses.

## Module map

| Module | Responsibility |
| --- | --- |
| `advantage.py` | The estimator. Start here. |
| `config.py` | Typed view over the `selaur.*` Hydra section, with fraction parsing and the deprecated `uqgigpo.*` alias. |
| `estimators.py` | Canonical `adv_estimator` names, shared by the trainer and the rollout collector. |
| `grouping.py` | Anchor-state grouping for the step level. |
| `normalization.py` | The shared group-baseline operation. |
| `returns.py` | Discounted step returns and the batch success rate. |
| `typedefs.py` | Shape and type conventions used throughout. |
| `uncertainty/signals.py` | Per-step uncertainty signals. |
| `uncertainty/aggregation.py` | Backward discounted smoothing along a trajectory. |
| `uncertainty/transforms.py` | Shaping transforms and the outcome bucket each owns. |
| `uncertainty/estimator.py` | Runs the above over a batch. |
| `baselines/gigpo.py` | GiGPO, expressed as SELAUR with shaping switched off. |

## Baseline parity

`baselines/gigpo.py` does not reimplement GiGPO. It calls the same estimator
with `apply_uncertainty_shaping=False`, so the two arms of every comparison
share one code path and an improvement to the grouping or the normalisation
cannot silently apply to only one of them. The baseline still computes and logs
uncertainty diagnostics, which is what makes the W&B curves directly
comparable.

## Extending it

* **A new uncertainty signal** — add a function to `uncertainty/signals.py`
  and register it in `SIGNAL_REGISTRY`. It becomes available as
  `selaur.method=<name>` (with `selaur.use_combined=False`).
* **A new shaping rule** — add it to `uncertainty/transforms.py`, list it in
  either `ADDITIVE_TRANSFORMS` or `MULTIPLICATIVE_TRANSFORMS`, and handle it in
  both `apply_step_transform` and `apply_episode_transform`. `advantage.py`
  dispatches on the family, so nothing there needs to change.
* **A new config knob** — add a field to `SELAURConfig` and a line to the
  `selaur:` block in `verl/trainer/config/ppo_trainer.yaml`. Unknown keys are
  rejected at startup, so the two cannot drift apart unnoticed.

Tests live in `tests/selaur/`.
