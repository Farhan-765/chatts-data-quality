"""
MCQ prompt templates for ChatTS anomaly classification.
Three templates covering different diagnostic scenarios.
"""

MCQ_CATEGORIES = """
This is an industrial sensor measurement sampled every 15 minutes.
Examine the baseline (normal operating reference line), signal envelope
(band of normal variation), and any departures from expected behavior.

Choose the most accurate description:

A) Drift - also called: baseline drift, reference line creep, calibration aging,
   setpoint deviation, or slow baseline migration.
   The reference line slowly moves upward or downward over weeks or months.
   The signal still oscillates normally but around a slowly moving center.
   The floor and ceiling of normal variation both creep together in same direction.
   Test: mean of first 30 points vs mean of last 30 points differ by > 3x noise,
   with change happening gradually across the full series, not abruptly.

B) Spikes - also called: outliers, excursions, transients, glitches, impulse noise,
   or momentary envelope violations.
   Sudden extreme jumps or drops far outside the normal signal envelope,
   returning to the baseline reference line within 1-3 points.
   The baseline and steady-state value are unaffected before and after each spike.
   Test: one or more points exceed 3 standard deviations from the local rolling mean.

C) Frozen segment - also called: stale data, flatline, signal freeze, stuck value,
   steady-state lock, dead band lock, zero-variation period, or sensor flatline.
   The signal stops at a fixed steady-state value with zero variation.
   The floor and ceiling of the envelope collapse to a single flat line.
   Normal oscillation resumes after the frozen period ends.
   Test: std of middle section drops to near zero (< 0.01) while
   std of first and last sections remain above 0.1.

D) Phase Change - also called: level shift, step change, baseline jump,
   setpoint offset, reference line displacement, or permanent floor shift.
   The baseline reference line shifts abruptly to a new permanent level.
   Unlike drift: shift is sudden, not gradual over weeks.
   Unlike spikes: the signal does NOT return to the original baseline.
   Test: mean of last 30 points differs from mean of first 30 points by > 5x noise,
   AND the transition completes within fewer than 5 consecutive points.

E) None - clean data. Baseline reference line is stable.
   Signal envelope (normal min/max band) is consistent throughout.
   Floor and ceiling of variation remain at expected levels.
   Daily/weekly cycles are normal process behavior, not anomalies.
   Test: std consistent across all sections, no flat regions, no extreme outliers.

G) Variance collapse - amplitude intermittently collapses to near-zero
   but signal does NOT completely stop (values still change slightly).
   Also called: amplitude dampening, envelope collapse, signal compression.
   Different from C (Frozen): values still change, just with very tiny amplitude.
   Test: rolling 10pt std drops near-zero repeatedly (more than 100 windows below 0.01)
   but consecutive value differences are NOT zero.

L) Intermittent failure - sensor produces physically impossible values.
   Examples: negative TOC, negative dissolved oxygen, negative ammonia.
   These violate physical laws - they are not just large outliers.
   The sensor briefly malfunctions then recovers to normal readings.
   Test: values below the physical minimum for this measurement type.

Answer with the letter (A/B/C/D/E/G/L), the category name, and a one-sentence explanation.
Include std values for first/middle/last 30 points to support your answer.
"""


MCQ_MULTI = """
This is an industrial sensor measurement sampled every 15 minutes.
Examine the baseline reference line, signal envelope, floor/ceiling levels,
and any departures from expected steady-state behavior.
Identify ALL data quality issues present:

A) Drift - baseline reference line slowly creeping over weeks/months.
   Also called: calibration drift, setpoint deviation, reference line migration.
   Floor and ceiling both shift gradually together.
   Test: mean(first 30) vs mean(last 30) differ by > 3x noise, change is gradual.

B) Spikes - momentary excursions outside the normal signal envelope.
   Also called: outliers, transients, glitches, impulse noise, envelope violations.
   Each excursion lasts 1-3 points then returns to baseline reference line.
   Test: any point exceeds 3 standard deviations from the local rolling mean.

C) Frozen segment - signal flatlines at a fixed steady-state value (std = 0).
   Also called: stale data, flatline, signal freeze, dead band lock,
   stuck value, zero-variation period, floor lock, or sensor flatline.
   Signal envelope collapses - floor and ceiling become identical.
   Normal envelope resumes after the frozen period ends.
   Test: std of any 10-point window drops to near zero (< 0.01)
   while surrounding windows maintain std > 0.1.

D) Phase Change - baseline reference line jumps to a new permanent level.
   Also called: level shift, step change, setpoint offset, floor shift,
   reference line displacement, or permanent baseline relocation.
   Entire signal envelope relocates abruptly - new floor and ceiling stabilize.
   Unlike spikes: no return to original baseline.
   Unlike drift: transition completes in < 5 points, not gradually.
   Test: mean shifts by > 5x noise within fewer than 5 consecutive points, permanently.

E) None - clean data. Stable reference line, consistent signal envelope,
   steady floor and ceiling levels, no excursions, no flatlines, no baseline migration.

G) Variance collapse - amplitude intermittently collapses to near-zero
   but signal does NOT completely stop (values still change slightly).
   Different from C: consecutive diffs != 0, signal still moves slightly.
   Test: rolling 10pt std drops near-zero repeatedly but no truly identical readings.

L) Intermittent failure - sensor produces physically impossible values.
   Examples: negative TOC, negative dissolved oxygen, negative ammonia.
   Test: values below the physical minimum for this measurement type.

For each issue found state: letter, name, point range, and numerical evidence.
If none apply, state E) None.
"""


MCQ_STALE_FOCUSED = """
This is an industrial sensor measurement sampled every 15 minutes.
Focus on whether the signal envelope collapses at any point.

A real sensor always produces a signal envelope with non-zero width.
If the floor and ceiling become identical (flatline, zero envelope width),
this is a frozen segment regardless of what value the signal is frozen at.

Step 1 - Calculate standard deviation of:
  std_start = std of first 30 points  (normal envelope width)
  std_mid   = std of middle 30 points (check if envelope collapses here)
  std_end   = std of last 30 points   (check if envelope recovers here)

Step 2 - Choose from:
A) Drift          - std consistent throughout, but mean shifts gradually.
   Also called: baseline drift, calibration aging, setpoint deviation.

B) Spikes         - std consistent, but 1-3 extreme points violate envelope boundary.
   Also called: outliers, excursions, transients, glitches, impulse noise.

C) Frozen segment - std_mid near zero (envelope collapses to flatline),
   while std_start and std_end show normal envelope width > 0.1.
   Also called: stale data, flatline, signal freeze, stuck value,
   steady-state lock, dead band lock, zero-variation period,
   floor lock, reference line collapse, or sensor flatline.
   The signal envelope disappears then reappears.
   Floor = ceiling during the frozen period - perfectly flat steady-state value.

D) Phase Change   - std consistent throughout, but mean jumps abruptly.
   Also called: level shift, step change, baseline jump, floor shift.

E) None           - std consistent, mean stable, no outliers, no flatlines.

G) Variance collapse - std drops near-zero in some windows but not all,
   and consecutive readings still differ (not truly frozen).
   The amplitude shrinks dramatically in sections then recovers.

Decision rule: if std_mid < 0.01 AND std_start > 0.1 AND std_end > 0.1 → answer is C.
               if many windows have std < 0.01 but zero_diffs = 0 → answer is G.
Answer with the letter. Show your std_start, std_mid, std_end calculations.
"""

# Alias used throughout the notebook
MCQ_EXTENDED = MCQ_CATEGORIES
