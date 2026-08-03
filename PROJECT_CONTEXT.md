# PROJECT_CONTEXT.md

Handoff document. Read this first.
Last updated: 3 August 2026 (§7 items 1–4 executed; **Finding 4 retracted**, see §3).

---

## 1. What this project is

A Master's dissertation replicating **Pan, Shang, Liu, Shao, Guo, Ding & Hu (2024),
"Spatial–Temporal Attention Network for Depression Recognition from facial videos,"
Expert Systems With Applications 237:121410.**

The task is BDI-II severity regression (0–63) from facial video. The paper's contribution is
the STA module, which replaces the middle 3x3x3 convolution in each 3D-ResNet bottleneck with
two parallel branches — a spatial branch pooling over time, a temporal branch pooling over
space — whose attention vectors are combined multiplicatively (attention vector-wise fusion)
rather than summed.

Paper's reported results: **AVEC 2013 MAE 6.15 / RMSE 7.98; AVEC 2014 MAE 6.00 / RMSE 7.75.**

**Agreed aim with supervisor:** replicate the results as far as available resources permit,
given a substantial disadvantage in hardware and dataset access. Nothing more was formally
agreed. Earlier planning documents (PLAN.md, HYPOTHESES.md) proposing extensions A1/A2/B1/B2
were drafted by the assistant and were never ratified — treat them as historical, not as
commitments.

**Dissertation title (locked):**
*Within the Noise: A Replication Study of Depression Severity Estimation from Facial Video*

**Framing (locked):** replication + critical audit. The result is a negative replication and
is written up as a finding, not a shortfall.

---

## 2. Current results

Best run: **fkvltxb3** (`f32_multistep_ms15-30`)

| Metric | Value |
|---|---|
| best val MAE | 9.167 |
| final val MAE | 9.305 |
| final val RMSE | 10.95 |
| epochs | 80 (early stopped) |

13 runs logged to W&B (`shivam03ray-newcastle-university/STA-DRN-II`), 12 finished, 1 crashed.
All use GroupNorm(8). Spread across the 12 finished runs: 9.167 to 10.463.

Re-scoring the saved `best.pth` for this run on 3 Aug 2026 reproduced **MAE 9.1672** exactly.
Its RMSE is **11.030**; the 10.95 above is the *final-epoch* RMSE, a different quantity. Quote
9.167 / 11.030 together as the best checkpoint's numbers — do not mix the two rows.

---

## 3. The findings

Four live (1, 2, 3, 4'), one retracted (the original 4). Read the retraction before citing
anything from an earlier draft of this file.

### Finding 1 — run-to-run variance exceeds the hyperparameter sweep

Two runs of an **identical configuration** (multistep milestones 20/40, features 32, lr 1e-4,
sigma 0, frame_len 64, GroupNorm, Adam):

- `avec_features32_gn` (results_log.csv, 21 Jul) → **9.2421**
- `f32_multistep_baseline` / 27gikqw2 → **10.2987**

**Variance on a fixed config = 1.06 MAE.** Total spread across all 12 configs = 1.296.
The noise is **82% of the entire observed range.** The headline improvement of ms15-30 over
baseline (1.13) is smaller than the noise on a single config.

Supporting: mean gap between best and final val MAE across runs = 0.044, so the curves are
flat at the end. The optimism is in *config selection*, not best-epoch cherry-picking.

### Finding 2 — the model barely beats a constant

Labels reconstructed from `dataset/avec14.csv`:
train mean 15.02, median 12.5, sd 12.27; test mean 14.50, median 13.0, sd 11.54.

| Predictor | Test MAE | Test RMSE |
|---|---|---|
| Constant = training mean (15.02) | 9.62 | 11.49 |
| Constant = training median (12.50) | **9.54** | 11.65 |
| Best possible constant (oracle, 12.0) | 9.54 | — |
| **Best model (fkvltxb3)** | **9.17** | **10.95** |
| Paper, STA-18 | 6.00 | 7.75 |

Margin over the strongest possible constant: **0.37 MAE.** Only 5 of 12 finished runs beat
it at all. Implied R² on the test set ≈ **0.09**.

**Strengthened 3 Aug 2026 by a paired bootstrap** (10,000 resamples, from `preds_ms15-30.csv`):

| comparison | Δ MAE | 95% CI | verdict |
|---|---|---|---|
| model − constant 12.50 | −0.373 | [−0.862, **+0.131**] | **not significant** |
| model − constant 15.02 | −0.457 | [−1.006, **+0.112**] | **not significant** |

Both intervals cross zero. The margin over a constant is **not statistically distinguishable
from zero**. This is stronger than "barely beats a constant" and should be the claim made.

Full metrics for the best run: MAE 9.167 [7.963, 10.390], RMSE 11.030 [9.744, 12.332],
PCC 0.304 [0.141, 0.455], CCC 0.130 [0.060, 0.194], R² 0.077. PCC and CCC *do* exclude zero,
so the model has learned something — it is just far too little to beat a constant on MAE.

The mechanism is visible in the per-band breakdown and is textbook regression to the mean.
**Predicted sd is 2.60 against a label sd of 11.48**; every prediction falls in [7.3, 20.1]
on a 0–63 scale.

| band | n | MAE | bias | mean pred |
|---|---|---|---|---|
| minimal 0–13 | 50 | 7.93 | **+7.60** | 12.56 |
| mild 14–19 | 20 | 3.88 | −2.39 | 13.81 |
| moderate 20–28 | 16 | 11.15 | −11.15 | 13.85 |
| severe 29–63 | 14 | 18.87 | **−18.87** | 15.27 |

The model predicts ≈13 for everyone. On the 14 most severe cases it is wrong by 18.9 points
on average. That is the clinically important sentence and it belongs in the abstract.

### Finding 3 — participants appear in both partitions

Ten participants appear in both train and test: **203, 214, 226, 234, 236, 237, 240, 242,
310, 317.** That is 24% of test participants and 20 of the 100 test recordings.

Derived from the numeric prefix of the folder names in `dataset/avec14.csv`. The
interpretation is independently corroborated: this gives 41 train / 42 test distinct
participants, and a published AVEC 2014 paper reports 41/43/42 across train/dev/test.

**IMPORTANT — this is not novel.** Lopez-Otero et al. documented speaker overlap in AVEC 2014
and the resulting bias; the speech literature responded by moving to leave-one-speaker-out.
The defensible claim is that **this has not propagated to the facial-video literature**, where
Pan et al. and every method in their comparison table still use the official partitions.
Do not claim discovery.

### Finding 4 — RETRACTED as originally stated; replaced by the checkpoint's actual score

**The original claim ("the published code cannot load its own checkpoint") is false. It was
tested on 3 Aug 2026 and the strict load succeeds. Do not put it in the dissertation.**

The error: `ResNet.__init__` builds each stage with `layers[i] * k` (`tsstanet.py:101-104`),
and `k=2`. So `layers=[2,2,2,2]` produces **4 blocks per stage, 16 total** — exactly what the
checkpoint contains. "4 blocks per stage" and "`layers=[4,4,4,4]`" are not the same thing; the
earlier audit conflated them. Verified exhaustively:

| construction | tensors | strict load |
|---|---|---|
| `layers=[2,2,2,2], k=2` | 586 | **succeeds** |
| `layers=[2,2,2,2], k=1` | 306 | mismatch |
| `layers=[4,4,4,4], k=1` | 586 | mismatch |
| `layers=[4,4,4,4], k=2` | 1146 | mismatch |

So the code, the checkpoint, and the paper's Table 1 (`x 2` per module) **agree with each
other**. `test_update.py:27` has always instantiated `layers=[2,2,2,2], k=2, features=16` and
loads `weights/best.pth` without error.

Still true and still verified: 586 tensors; **7,840,149** total (7,822,785 parameters +
17,364 BatchNorm buffers); BatchNorm; base width 16.

**What survives** is the parameter count only: the paper's Table 4 reports STA-18 at 31.27M,
this file holds 7.84M. That is a paper-vs-artefact discrepancy, not a code-vs-artefact one, and
it rests on the Table 4 figure — re-read the table before relying on it. It is a much weaker
claim than the retracted one and should not carry a section on its own.

### Finding 4' (new, strong) — the authors' published checkpoint scores 10.44, not 6.00

Run on 3 Aug 2026 under the authors' **own** published test configuration
(`layers=[2,2,2,2], k=2, features=16`, BatchNorm, `frame_len=64`, `SAMPLE_INTERVAL=3`,
their `transforms3d` preprocessing unmodified), on the official AVEC 2014 test partition:

| | MAE | RMSE | PCC | CCC | R² |
|---|---|---|---|---|---|
| **authors' `best.pth`** | **10.44** [9.14, 11.84] | 12.53 | 0.150 [−0.075, 0.352] | 0.126 [−0.058, 0.290] | **−0.19** |
| paper's claim | 6.00 | 7.75 | — | — | — |
| best constant (12.5) | 9.54 | 11.65 | 0 | 0 | −0.01 |

Two things matter more than the MAE gap: **R² is negative** (worse than predicting the mean),
and the **PCC and CCC confidence intervals both include zero** — the checkpoint's output is not
statistically distinguishable from being uncorrelated with the labels.

Robustness checks all done, none of them move it:
- averaging scheme — flat per-clip mean 10.443, mean-of-batch-means at their `BATCHSIZE=10`
  10.443, at our 2 10.462. Immaterial.
- `frame_len` 64 → 10.443, `frame_len` 128 → 10.481. Immaterial.

**Caveats that must be stated with this number:**
1. The corpus this checkpoint was trained on is unknown. It may not be the AVEC 2014 model.
2. The face crops are from *our* alignment pipeline, not the authors'. They are genuine
   224x224 aligned crops of the right subject (spot-checked visually) and the normalisation
   is the authors' own code, but the aligner is not theirs.
3. See §4a — over half our videos are too short to form a genuine clip.

Caveat 1 is the serious one and cannot be closed from here. Frame it as "the released
checkpoint does not reproduce the released number," not "the paper's number is wrong."

The article carries a Code Ocean "certified Reproducible" badge. State this carefully — the
badge may have certified a different entry point.

---

## 4. Known methodological problem in our own pipeline

`dataset/avec14.csv` has 200 rows: `train/` then `test/`. **There is no `dev/` partition in it.**
`train_update.py` splits `[:100]` train, `[100:200]` validation.

**So what we have called "validation" is the AVEC 2014 test partition.** Early stopping,
best-checkpoint saving, and selection of the winning config out of 12 were all done against
the set we report on.

Consequence: 9.17 is on the same footing as the paper's 6.00 (both test-set numbers), but it is
a best-of-12 selected on the reported set and is optimistically biased by an unknown amount.
Under a null where all configs are identical, best-of-12 would drift ~1.2 MAE below truth by
chance alone.

Three honest routes: (a) extract the real AVEC 2014 dev partition and redo selection on it;
(b) carve a dev split from the 100 training videos and leave test untouched; (c) report as-is
with the bias stated plainly as a limitation. Route (b) preferred — costs one retrain.

**Corroborated 3 Aug 2026.** The authors' original `test_update.py` (commit `fadef81`, before
any of our edits) slices `image_path_list[200:]`, and their `test.py:59` slices `[200:300]`.
Both imply a **300-row** CSV — 100 train / 100 dev / 100 test. Ours has 200 rows and is missing
the dev partition entirely, which is exactly why our `[100:200]` lands on test. Our partition
composition is otherwise correct and official: 100 videos each, 50 Freeform + 50 Northwind.

## 4a. Second methodological problem — over half the videos cannot form a real clip

Found 3 Aug 2026, not previously recorded. Frame counts per video are wildly uneven:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| train | 14 | 149 | 176 | 258 | 920 |
| test | 35 | 168 | 198 | 637 | 4701 |

After `SAMPLE_INTERVAL=3`, a video needs 192 raw frames to fill one 64-frame clip. **108 of
200 videos (61 train, 47 test) fall short.** Those hit the `len(image_name) <= frame_len`
branch in `main_dataloader.py:148-150`, which index-stretches the available frames to length
64 — i.e. feeds the network **repeated frames**. 14 videos have under 32 effective frames;
3 have under 10.

Why it matters: the paper's entire contribution is a *temporal* attention branch. On more than
half the corpus we are handing that branch a clip whose temporal variation is largely
interpolation artefact. This is an independent candidate explanation for the gap, alongside
`PRETRAIN = False`, and it is our pipeline's fault rather than the method's.

It is **not** confounded with severity, which is the good news — mean label is 14.83 for the
degenerate videos vs 14.67 for the full-length ones, Pearson r(log frames, BDI-II) = −0.138,
Spearman ρ = −0.056. So it injects noise, not bias. Say that explicitly, because a reader will
otherwise assume the short videos are the severe cases.

Fix if there is time: re-extract frames, or drop to `SAMPLE_INTERVAL=1` for short videos.

---

## 5. Deviations from the paper — all must be documented

| Aspect | Paper | This work |
|---|---|---|
| Pretraining | UCF101 (temporal) + CK+ (spatial) | **`PRETRAIN = False`** — trained from scratch |
| Training corpora | AVEC 2013 + 2014 union (150 videos) | AVEC 2014 only (100 videos) |
| Frame sampling | paper text says interval 1 | `SAMPLE_INTERVAL = 3` — **but so does their code** (see below) |
| Base width | 16 | 32 (best run) |
| Normalisation | BatchNorm | GroupNorm(8) |
| Hardware | 6x TITAN RTX | single 8 GB GPU |
| Batch size | 5 | 2, with gradient accumulation |
| Seeds | not stated | **none ever set** |

`PRETRAIN = False` is the most likely single cause of the 9.17 vs 6.00 gap and should be
named as such. §4a is now a second candidate.

**Correction (3 Aug 2026) to the frame-sampling row.** `SAMPLE_INTERVAL = 3` is **not our
deviation** — it is the authors' own published default. Verified in the original upload
(`fadef81`): `test_update.py:16` sets `SAMPLE_INTERVAL = 3`, and `main_dataloader.py` declares
`sample_interval=3` as the default on both `MainDataset` and `ValDataset`. If the paper's text
says interval 1, that is a discrepancy *internal to the authors' release*, and we inherited
their code's value. Do not list it in our deviations table — check the paper text and, if it
does say 1, move it to Finding 4's paper-vs-artefact list where it belongs.

Also corrected: `frame_len=128` in the current `test_update.py:18` is **our** edit. The
authors' original had `frame_len = 64` (`fadef81:17`), matching training. Scoring at 128 costs
about 0.04 MAE on the published checkpoint, so nothing rests on it, but the deviation is ours.

### Why GroupNorm

An early run produced val MAE 210 / RMSE 773. Diagnosed as BatchNorm running statistics
poisoned by `batch_size=1` combined with an unbounded `relu(pred) * SCORE_RANGE` output.
GroupNorm(8) was adopted as the fix. This is a justified deviation, not an arbitrary one.

---

## 6. Writing status

- **Title** — locked (§1 above)
- **Abstract** — 189 words drafted, conclusion sentence deliberately left open pending the
  pretraining experiment
- **Introduction §1.1–1.4** — drafted, 1,462 words, in `introduction_draft_v1.md`
- **Introduction §1.5 (Aims and Objectives)** — not written; must state the aim as agreed at
  the outset (see §1 above) and must NOT be revised in light of the outcome
- Unmet objectives from PLAN.md belong in the Discussion, not the Introduction

### References verified so far

- Musgrave, Belongie & Lim, *A Metric Learning Reality Check*, ECCV 2020, pp. 681–699
- Bouthillier et al., *Accounting for Variance in ML Benchmarks*, MLSys 2021
- DeMasi, Kording & Recht, *Meaningless comparisons lead to false optimism in medical machine
  learning*, PLOS ONE 12(9):e0184604, 2017
- Sculley, Snoek, Wiltschko & Rahimi, *Winner's Curse? On Pace, Progress, and Empirical Rigor*,
  2018 (venue needs confirming — ICLR workshop)

Several references in `introduction_draft_v1.md` are flagged INCOMPLETE and must be located
and verified by the author before submission. The Lopez-Otero reference is the most important:
Finding 3's framing rests on it and it is currently known only second-hand.

---

## 7. Outstanding work

Ordered by value per minute.

**Items 1–4 were completed on 3 August 2026.** Results folded into §3 and §4a above.

1. ~~**Verify the label reconstruction**~~ — **DONE.** Returns exactly `200 15.02 13.0`.
   Finding 2's label basis is confirmed.

2. ~~**Capture the strict-load failure**~~ — **DONE, and it did not reproduce.** The strict
   load *succeeds*. Finding 4 is retracted; see §3. This item is closed, not outstanding.

3. ~~**Score the authors' published checkpoint**~~ — **DONE. MAE 10.44 / RMSE 12.53**, against
   the paper's 6.00 / 7.75, with negative R² and PCC/CCC intervals spanning zero. This is the
   new Finding 4' in §3. Both caveats the item asked for were discharged: the averaging scheme
   was checked line-by-line against the original `test_update.py` (three variants, max spread
   0.02 MAE), and the unknown-training-corpus caveat is recorded in §3 as the limiting one.
   Note the command printed in earlier versions of this file was wrong — `LAYERS=4,4,4,4`
   builds eight blocks per stage and aborts. The correct invocation is:
   ```bash
   CKPT=weights/best.pth LAYERS=2,2,2,2 FEATURES=16 NORM=batchnorm \
   OUT=preds_authors.csv python dump_predictions.py
   ```

4. ~~**Dump predictions for the best run**~~ — **DONE.** `preds_ms15-30.csv`, reproducing
   MAE 9.1672 against the logged 9.167. PCC/CCC/CIs/per-band errors are in §3.
   ```bash
   CKPT=weights/f32_multistep_ms15-30/best.pth OUT=preds_ms15-30.csv python dump_predictions.py
   ```

**Now the top of the list:**

5. **Pretraining run, 2 seeds** (~3 h GPU) — the single highest-value experiment remaining.
   Without it, "the method fails" cannot be separated from "our initialisation fails."
   **Two seeds minimum.** Given Finding 1, one run yields a number that cannot be interpreted;
   running one would repeat the exact error the dissertation criticises.

6. **Subject-disjoint split + 2 retrains** (~30 min scripting, ~3 h GPU) — converts Finding 3
   from a metadata audit into an experimental result.

Items 1–4 are done and the results chapter is unblocked. Item 5 gates the abstract's final
sentence and is now the top priority; item 7 below is new.

7. **Re-extract frames, or make sampling adaptive** (~1 h scripting + retrain) — addresses
   §4a. Lower value than 5 and 6 for the dissertation's argument, but it is the only item that
   could move our own MAE materially, and it is a defect a marker may well spot unaided.

---

## 8. Figure to build

The twelve runs plotted against the constant-predictor line at 9.54, with the replicate pair
(9.24 / 10.30) marked. This is the image that makes the title literal and should appear early
in the results chapter, not buried.
