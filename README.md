# Baseline_of_Plan2Align

Baseline implementations used for comparison in **Plan2Align / TMPC** — test-time
preference alignment for large language models.

> 📄 **Paper:** [Test-Time Alignment for Large Language Models via Textual Model Predictive Control](https://arxiv.org/abs/2502.20795) — **ICLR 2026**
>
> Kuang-Da Wang, **Teng-Ruei Chen**, Yu Heng Hung, Guo-Xun Ko, Shuoyang Ding,
> Kris Wu, Yu-Chiang Frank Wang, Huck Yang, Wen-Chih Peng, Ping-Chun Hsieh
>
> 🔗 **Main method:** [Plan2Align](https://github.com/ray24724919/Plan2Align) · **Official implementation:** [rl-bandits-lab/Plan2Align](https://github.com/rl-bandits-lab/Plan2Align)

Plan2Align aligns at inference time by planning over segments. To show the gain comes
from *planning* and not simply from spending more test-time compute, it has to be
measured against other decoding-time alignment methods under the same reward model and
the same evaluation metric. That is what this repository holds.

## Baselines

| Path | Method |
|---|---|
| `args/` | **ARGS** — Auto-Regressive Guided Search: reward-guided token-level decoding, run on HelpSteer |
| `args_translation/` | **ARGS** for paragraph-level translation (zh→en / zh→de / zh→ru), scored with MetricX-24 |
| `rain/` | **RAIN** — Rewindable Auto-regressive INference: self-evaluation and rewind inside a single decoding pass |
| `metricx24/` | **MetricX-24** — reference-free translation quality metric, the shared scoring harness for the translation comparisons |

ARGS steers each token by reward. RAIN searches and rewinds within one pass. Both act at
decoding time, but neither scores what a *segment* contributes to the finished response
— which is the axis the comparison is meant to isolate.

Defaults across the baselines match the main method: `meta-llama/Meta-Llama-3.1-8B-Instruct`
as the language model, `rl-bandits-lab/helpsteer_rm` as the reward model, 1024 max new tokens.

## Setup

```bash
conda env create -f environment.yml
conda activate p2a
```

## Running

```bash
# ARGS on HelpSteer
python args/main.py --dataset <path/to/helpsteer_test.csv> \
                    --llm_gpu cuda:0 --rm_gpu cuda:1 --out_file args/run_outs

# ARGS for translation
python args_translation/main.py --language en --type paragraph

# RAIN
python rain/main.py --dataset <path/to/helpsteer_test.csv>

# Score translation outputs with MetricX-24
python -m metricx24.predict --input_file <run_outs.jsonl> --output_file <scores.jsonl>
```

Datasets are not included — point `--dataset` at the HelpSteer / ACL-2025 translation
files, or take them from the official repository. Each script's remaining flags are at
the top of its `main.py`.

> The `--dataset` defaults still point at the lab machine these were run on
> (`/home/raychen/...`). Pass the flag explicitly.

## Citation

```bibtex
@inproceedings{wang2026testtime,
  title     = {Test-Time Alignment for Large Language Models via Textual Model Predictive Control},
  author    = {Wang, Kuang-Da and Chen, Teng-Ruei and Hung, Yu Heng and Ko, Guo-Xun and
               Ding, Shuoyang and Wu, Kris and Wang, Yu-Chiang Frank and Yang, Huck and
               Peng, Wen-Chih and Hsieh, Ping-Chun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2502.20795}
}
```
