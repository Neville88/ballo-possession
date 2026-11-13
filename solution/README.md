# solution/

Team ballo's entry for More than a Hack 2027, Phase 1. Nothing is trained or tuned on any ground truth, public or private. See
[TECHNICAL.md](TECHNICAL.md) for the method, how it was chosen and its limitations.

## Reproduce the submission

```bash
pip install -r solution/requirements.txt
python solution/run.py --team ballo            
python solution/run.py --team ballo --check   
```

Writes `evaluation/results/ballo.csv`, validates it with the organisers' validator and prints its
SHA-256:

```
06d72eb684cab9f7d09ab084e268c782c7d528fb1ca62751f7aa9dad71684da3
```

The match data is not in this repository. Unpack it into `data/` first (see
[DATA.md](../DATA.md)).

## Files

| file | what it does |
|---|---|
| `run.py` | the one command: build, validate, hash, optional reproducibility check |
| `possession.py` | the model: per-frame evidence for D / H / A, a switch-penalised Viterbi path, then tidying |
| `deadball.py` | dead-ball evidence: naive-Bayes log ratio over five tracking signals |
| `pseudo.py` | reference windows from the automatic Smart Tagging, used to compare versions |
| `teammodel.py` | attack direction, and a team-shape possession model (built, measured, weighted 0) |
| `features.py` | parses the tracking once and caches it under `data/cache/` (gitignored) |
| `selfcheck.py` | full-match, ground-truth-free check of the current model |
| `compare.py` | the same check for any set of CSVs |
| `diagnostics.py` | which signals separate stoppages from play, as percentiles |
| `review.py` | lists a submission's possessions and stoppages with video times, for checking by eye |
| `TECHNICAL.md` | the technical document |

## Comparing submissions

Put every CSV in `evaluation/results/`, then:

```bash
python evaluation/run_evaluation.py     # public 5 minutes: is each pipeline correct?
python solution/compare.py              # full match vs Smart Tagging: which is better?
```

The public window has 17 possessions and ranks two baselines in the wrong order (see
[EVALUATION.md](../EVALUATION.md)), so it checks correctness, not rank. `compare.py` covers all 106
minutes. Gaps of a few points on either are ties.

## Checking against the video

```bash
python solution/review.py                                  # reviews ballo
python solution/review.py evaluation/results/<name>.csv
```

Writes `evaluation/review/<name>_segments.csv`: every possession and stoppage with its start and
end as `mm:ss.ff` in `data/tactical/BAR-ATH_tactical.mp4`, with minutes running on past 60. The
algorithm does not read the video. See [TECHNICAL.md](TECHNICAL.md) for how watching it was used.

## Data safety

`data/`, the tracking cache, every CSV in `evaluation/results/` except the organisers' baselines,
`evaluation/review/` and `evaluation/reports/` are all gitignored because they are, or are derived from,
the licensed match data and must never be published.
