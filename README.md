# Baseline_of_Plan2Align

Baseline implementations used for comparison in **Plan2Align / TMPC** — test-time
preference alignment for large language models.

> 📄 **Paper:** [Test-Time Alignment for Large Language Models via Textual Model Predictive Control](https://arxiv.org/abs/2502.20795) — **ICLR 2026**
>
> Kuang-Da Wang, **Teng-Ruei Chen**, Yu Heng Hung, Guo-Xun Ko, Shuoyang Ding,
> Kris Wu, Yu-Chiang Frank Wang, Huck Yang, Wen-Chih Peng, Ping-Chun Hsieh
>
> 🔗 **Main method:** [Plan2Align](https://github.com/ray24724919/Plan2Align) · **Official implementation:** [rl-bandits-lab/Plan2Align](https://github.com/rl-bandits-lab/Plan2Align)

Plan2Align aligns at inference time by planning over segments. To show that the gain
comes from *planning* rather than simply from spending more test-time compute, it has to
be measured against other decoding-time alignment methods under the same reward model
and the same evaluation metric. That is what this repository holds.

## Baselines

| Path | Method |
|---|---|
| `args/` | **ARGS** — Auto-Regressive Guided Search: reward-guided token-level decoding, run on helpfulness data |
| `args_translation/` | **ARGS** adapted to paragraph-level translation, scored with MetricX-24 / COMET |
| `rain/` | **RAIN** — Rewindable Auto-regressive INference: MCTS-style self-evaluation and rewind during decoding |
| `metricx24/` | **MetricX-24** — Google's reference-free translation quality metric, used as the shared evaluation harness |

ARGS steers each token by reward; RAIN searches and rewinds within a single pass. Both
act at decoding time but neither plans over segment-level contributions the way
Plan2Align does — which is the axis the comparison is meant to isolate.

## Setup

```bash
conda env create -f environment.yml
```

Each baseline directory runs independently; see the scripts inside for the entry point
and arguments. `metricx24/predict.py` produces the quality scores that the translation
comparisons are reported on.
