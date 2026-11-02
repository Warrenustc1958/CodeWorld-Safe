# Within-group variance measurement and start-state selection (2026-11-02)

Follows `E12_E14_RM_PROGRESS.md`. Pre-registered design: `VARIANCE_PREREG.md` (written before the runs).
All numbers come from per-episode dumps with synchronised real-simulator labels; policy = base π0.5 frozen (lr 0).

## 1. Why calibrated RMs still fail in GRPO

Online (E15, 5120 episodes each): R-A fired falsely on 2.9% of failures but missed 38.6% of successes; R-C 0.7% / 60.2%.
Advantage cosine between RM-based and real-reward GRPO advantages: E15-A .18, E15-Q .17, E14 (random label noise,
no gain) .42, real reward 1.0. Offline threshold sweep: best cosine .34 (R-A at .4) / .26 (R-C); noise floor,
continuous reward and confidence gating reach at most .40.

## 2. Group outcomes are decided by the start state

| condition | success | m (mixed groups) | within-group variance share |
|---|---|---|---|
| M0 mid keyframes, 64 steps, noise .5 | .401 | .164 | .123 |
| M1 initial-state keyframes (t0 = 8), 256 steps | .492 | .469 | .335 |
| M2 noise .8 | .407 | .180 | .129 |
| M3 noise 1.0 | .416 | .195 | .140 |

- Exploration noise barely changes m; the start state does.
- Test-retest under the same policy (254 keyframes drawn >= 2 times): corr of group success rate .976;
  P(mixed again | mixed) .75 vs .065.
- Selecting start states on a first draw raises m on held-out draws: real outcome .75, R-A group mean in [.2, .8] .47
  (no labels needed).
- Source-trajectory "decision moment" keyframes (8–48 steps before first success / violation) do not help:
  m .14–.23 vs .14–.17.

## 3. But the RM is worst exactly where the signal is

| selection (draw 1) | m | FP | FN | spurious / real | best cosine |
|---|---|---|---|---|---|
| none | .191 | .030 | .341 | .714 | .383 |
| real mixed (oracle) | .733 | .131 | .474 | .175 | .492 |
| R-A group mean in [.2, .8] | .474 | .109 | .404 | .889 | .559 |

No rule passes the pre-registered gate (m >= .40, cosine >= .60, spurious/real <= .5). Boundary states, whose outcome
is undecided, are also the hardest to judge from images: FP rises from 3% to 13%, FN from 34% to 47%.

## 4. Long horizons

M1 truncated at H chunks: FP .013 (H 8) → .117 (H 32), cosine .49 → .18 while m rises .20 → .47. Whether this is Wan
drift or the RM being out of distribution was to be settled by M1R (real rendered observations, same start states);
M1R and E16-L (initial-state 256-step real-reward GRPO) were stopped by another user at 15:06 (auth.log:
`systemctl kill/stop cws-m1r cws-e16l`), M1R has 496 episodes.

## 5. Corrections to earlier claims

- The independent-error formula for spurious groups over-predicts by 2–6x: RM errors are correlated within a group
  (the same scene is misjudged for all 8 rollouts). The "critical error rate ≈ 2.5%" statement is withdrawn.
- A threshold-sweep table was once reported before the script had run; the real output is in `rm2/thr_sweep.txt`.

## 6. Open directions (not yet validated)

1. On-policy hard-example RM training on exactly the boundary states that start-state selection picks.
2. Success judgement with state information (Wan state reader) in addition to images, at boundary states.
3. Measure Wan drift directly: render the real simulator at every chunk end during a Wan-observation run.
