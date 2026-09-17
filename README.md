<h1 align="center">SELAUR</h1>
<p align="center"><b>Self Evolving LLM Agent via Uncertainty-aware Rewards</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.21158"><img src="https://img.shields.io/badge/arXiv-2602.21158-b31b1b.svg?style=flat-square&logo=arxiv" alt="arXiv"></a>
  &nbsp;<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square" alt="License"></a>
</p>

<p align="center">Dengjia Zhang · Xiaoou Liu · Lu Cheng · Yaqing Wang · Kenton Murray · Hua Wei</p>

Official implementation of [**SELAUR**](https://arxiv.org/abs/2602.21158).

A reward tells you whether an agent reached the goal, not whether it *knew what
it was doing*. But the policy records its own conviction in every rollout — the
log-probabilities of the tokens it emitted. SELAUR turns that into a per-step
uncertainty estimate (entropy + least-confidence + margin) and injects it into
step- and trajectory-level rewards, on top of a critic-free two-level group
advantage.

<details>
<summary>Abstract</summary>

Large language models (LLMs) are increasingly deployed as multi-step
decision-making agents, where effective reward design is essential for guiding
learning. Although recent work explores various forms of reward shaping and
step-level credit assignment, a key signal remains largely overlooked: the
intrinsic uncertainty of LLMs. Uncertainty reflects model confidence, reveals
where exploration is needed, and offers valuable learning cues even in failed
trajectories. We introduce SELAUR: Self Evolving LLM Agent via Uncertainty-aware
Rewards, a reinforcement learning framework that incorporates uncertainty
directly into the reward design. SELAUR integrates entropy-, least-confidence-,
and margin-based metrics into a combined token-level uncertainty estimate,
providing dense confidence-aligned supervision, and employs a failure-aware
reward reshaping mechanism that injects these uncertainty signals into step- and
trajectory-level rewards to improve exploration efficiency and learning
stability. Experiments on two benchmarks, ALFWorld and WebShop, show that our
method consistently improves success rates over strong baselines. Ablation
studies further demonstrate how uncertainty signals enhance exploration and
robustness.
</details>

## Layout

```
selaur/           the algorithm — see selaur/README.md for the walkthrough
  advantage.py      uncertainty-aware two-level advantage (entry point)
  uncertainty/      signals · smoothing · transforms · estimator
  baselines/        GiGPO = SELAUR with shaping switched off
agent_system/     environments (ALFWorld, WebShop, …), multi-turn rollout
verl/             RL engine; `trainer/config/ppo_trainer.yaml` holds `selaur:`
examples/         launch scripts — selaur_trainer/ is the main one
tests/selaur/     unit tests for the algorithm
```

Paper → code: `entropy`/`least-confidence`/`margin` are `step_entropy`/
`step_nll`/`step_top2_margin` in `selaur/uncertainty/signals.py`; the combined
estimate is `combined_uncertainty`; failure-aware reshaping is in
`selaur/uncertainty/transforms.py` and `selaur/advantage.py`.

## Install

```bash
conda create -n selaur python==3.12 -y && conda activate selaur
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e . && pip3 install vllm==0.8.5
python -c "import verl, selaur, vllm; print('ok')"
```

Then install the agent environment — **each in its own conda env**, they pin
conflicting versions.

<details>
<summary>ALFWorld (works in the <code>selaur</code> env above)</summary>

```bash
pip3 install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f      # game files + MaskRCNN detector -> ~/.cache/alfworld/
alfworld-play-tw          # sanity check
```
</details>

<details>
<summary>WebShop (needs Python ≤ 3.10 — separate env)</summary>

```bash
conda create -n selaur-webshop python==3.10 -y && conda activate selaur-webshop
cd ./agent_system/environments/env_package/webshop/webshop && ./setup.sh -d all && cd -
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e . && pip3 install vllm==0.8.2    # 0.8.2, not 0.8.5
```
The `typer` warnings from spacy/weasel can be ignored.
</details>

Sokoban / gym_cards / AppWorld: see `agent_system/environments/README.md`.

## Train

```bash
conda activate selaur                          # selaur-webshop for WebShop
bash examples/selaur_trainer/run_webshop.sh    # Qwen2.5-1.5B-Instruct
bash examples/gigpo_trainer/run_webshop.sh     # GiGPO baseline, same budget
```

Each script prepares the data and trains in one go. Variants:
`run_{webshop,alfworld}{,_7b,_llama,_qwen3,_lora}.sh`. Reference wall-clock:
1.5B ≈ 12 h (WebShop) / 30 h (ALFWorld); 7B ≈ 40 h / 60 h.

The tunables sit at the top of each script (`group_size`, `transform`, `rho`,
`w1/w2/w3`, `step_weight`, `threshold_success_rate`, …). **Watch `group_size`**:
step-level groups are formed from identical anchor states, so if the average
group size the trainer prints hovers near 1, the step-level term is inert —
raise it.

Export a checkpoint:

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir ./checkpoints/webshop/selaur/global_step_150/actor \
    --target_dir ./checkpoints/out
```

## Configure

Set `algorithm.adv_estimator=selaur`, then tune `selaur.*`. Defaults and inline
comments: `verl/trainer/config/ppo_trainer.yaml`.

| Key | Default | Meaning |
| --- | --- | --- |
| `use_combined` | `True` | Blend all three metrics rather than using one. |
| `method` | `entropy` | Single metric when `use_combined=False`: `entropy`, `nll`, `top2`. |
| `w1` / `w2` / `w3` | `1/3` | Blend weights (entropy / least-confidence / margin). Fractions accepted. |
| `rho` | `1` | `1` = weighted average; `0` = whichever metric fired loudest. |
| `lambda_uq` | `0.95` | Backward discount when smoothing uncertainty along an episode. |
| `step_weight` | `0.85` | Step-level shaping strength (episode level is the `1.0` reference). |
| `transform` | `linear` | `linear`/`neg` reshape **failed** episodes; `exp` reshapes **successful** ones. |
| `shaping_scale` | `10.0` | Multiplier on the additive shaping term. |
| `threshold_success_rate` | `0` | Shaping stays off below this batch success rate. |

Both levels are reshaped in the *same* direction; the transform picks which.
Shared with GiGPO: `algorithm.gigpo.step_advantage_w`,
`algorithm.gigpo.mode` (`mean_norm` / `mean_std_norm`), `env.rollout.n`.

The old `adv_estimator=uqgigpo` and `uqgigpo.*` keys still work, with a
`DeprecationWarning`.

<details>
<summary>Troubleshooting</summary>

- **`Avg size of step-level group: 1.0x`** — raise `group_size` / `env.rollout.n`, or check the env emits deterministic anchor observations (grouping is exact matching).
- **Uncertainty metrics flat at zero** — shaping is gated off: the batch success rate has not passed `threshold_success_rate`. Diagnostics are logged either way, so flat *shaping* + live *diagnostics* means the gate.
- **CUDA OOM** — lower `rollout.gpu_memory_utilization` or `ppo_micro_batch_size_per_gpu`, or raise `tensor_model_parallel_size`.
- **`Unknown SELAUR config keys`** — a typo in a `selaur.*` override. Unknown keys are rejected at startup so a sweep cannot silently measure nothing.
</details>

## Test

```bash
pytest tests/selaur -q     # 94 tests
```

## Citation

```bibtex
@article{zhang2026selaur,
  title   = {SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards},
  author  = {Zhang, Dengjia and Liu, Xiaoou and Cheng, Lu and Wang, Yaqing and
             Murray, Kenton and Wei, Hua},
  journal = {arXiv preprint arXiv:2602.21158},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.21158}
}
```

## Acknowledgements

Built on [verl-agent / GiGPO](https://github.com/langfengQ/verl-agent) and
[veRL](https://github.com/volcengine/verl). Apache 2.0 — see `LICENSE` and
`Notice.txt`; the upstream verl-agent README is kept at
`docs/upstream_verl_agent.md`.
