#!/usr/bin/env bash
# CI guard: surfaces unjustified widenings of the Tier 3 baseline snapshot.
#
# The frozen-baseline policy in
# `crates/astrodyn_bevy/tests/README.md` "Baseline-freeze workflow" plus
# `crates/astrodyn_verif_jeod/src/bin/tier3_baseline_diff.rs` already
# guarantees that **refactor-only** PRs cannot regress past the
# committed `crates/astrodyn_verif_jeod/test_data/baselines.json`.
# That tool, however, compares the *current run* to the *frozen
# baseline file as it sits on disk in the PR branch* — so a PR that
# also edits `baselines.json` (a "physics change" refreeze) is
# self-consistent by construction and `tier3_baseline_diff` cannot
# detect that the new file widened a per-component error vs the
# version on `main`. The README mitigates this with a PR-description
# convention ("loosening a baseline requires a PR comment citing the
# physical justification"), but that mitigation depends on developer
# discipline.
#
# This script closes the loop structurally: it diffs the working
# `baselines.json` against `origin/main`'s version, computes a per
# component widening ratio for every (test, metric, component) tuple
# that exists in both versions, and emits GitHub Actions annotations
# bucketed by severity. Each bucket maps to one structural CI signal:
#
#   ratio ≤ WARN_RATIO_THRESHOLD     — silent (numerical drift is normal)
#   WARN < ratio ≤ FAIL_RATIO        — `::warning::` annotation, exit 0
#   ratio > FAIL_RATIO_THRESHOLD     — `::error::` annotation, exit 1
#
# Defaults: WARN=1.50 (50% widening), FAIL=2.00 (2× widening). Picked
# to match the README's existing per-component tolerance margin
# (`error * 1.05`) — a 50% widening is a multiple of that margin and
# warrants a deliberate refreeze, a 2× widening is large enough to
# block a PR until the regression or intentional change is reviewed
# in the PR description.
#
# What's not flagged:
#
#   - Brand-new test entries (no `main`-side value to widen against).
#   - Removed test entries (the README permits dropping tests when
#     they're retired upstream).
#   - Tightened entries (current < baseline-on-main — the desired
#     direction; surfaced as `::notice::` for visibility only).
#   - Entries where the `main`-side baseline is zero. A zero baseline
#     means JEOD's reference matches ours bit-for-bit on that channel,
#     and any non-zero new value would have an infinite ratio. The
#     `tier3_baseline_diff` `1e-12` absolute floor already gates these
#     against absolute drift; we omit them here rather than
#     double-counting.
#
# Special cases:
#
#   - Working tree identical to `origin/main`: silent OK.
#   - File added in the PR (no `main`-side version): silent OK
#     (first-freeze case; nothing to widen against).
#   - File removed in the PR: hard fail (the diff job downstream needs
#     the file).
#
# This script does **not** read `target/tier3_crossval/*.json`; that
# is `tier3_baseline_diff`'s job. The two checks compose: the runtime
# diff catches regressions within the *current* baseline, this static
# diff catches widenings of the baseline file itself.
set -euo pipefail

# Thresholds. Mirrored in `crates/astrodyn_bevy/tests/README.md`
# "Baseline-freeze workflow" — keep both in sync if you tune them.
WARN_RATIO=${BASELINE_WARN_RATIO:-1.50}
FAIL_RATIO=${BASELINE_FAIL_RATIO:-2.00}

# Allow callers to override the base ref (default: `origin/main`).
# Useful for local dry-runs against an arbitrary branch.
BASE_REF=${BASELINE_BASE_REF:-origin/main}

BASELINE_PATH=crates/astrodyn_verif_jeod/test_data/baselines.json

if [ ! -f "$BASELINE_PATH" ]; then
    echo "FAIL: $BASELINE_PATH is missing from the working tree." >&2
    echo "  This script and the downstream tier3_baseline_diff tool both need it." >&2
    exit 1
fi

# Fetch baseline contents from the base ref. If the base lacks the
# file (first-freeze on a new branch), exit silently with OK — there
# is nothing to widen against.
if ! base_raw=$(git show "${BASE_REF}:${BASELINE_PATH}" 2>/dev/null); then
    echo "OK: $BASELINE_PATH does not exist on ${BASE_REF} (first-freeze case); nothing to compare."
    exit 0
fi

current_raw=$(cat "$BASELINE_PATH")

# Fast path: file unchanged → silent OK without spinning up Python.
if [ "$base_raw" = "$current_raw" ]; then
    echo "OK: $BASELINE_PATH unchanged vs ${BASE_REF}; no widening to check."
    exit 0
fi

# Hand off to Python for the JSON walk. Python is the natural fit for
# nested JSON + arithmetic; bash + jq would need a much heavier
# wrapper to produce the same per-component annotations.
exec python3 - "$base_raw" "$current_raw" "$WARN_RATIO" "$FAIL_RATIO" "$BASELINE_PATH" "$BASE_REF" <<'PYEOF'
import json
import sys

base_raw, current_raw, warn_str, fail_str, baseline_path, base_ref = sys.argv[1:7]
WARN_RATIO = float(warn_str)
FAIL_RATIO = float(fail_str)

# Match `tier3_baseline_diff.rs::ABSOLUTE_FLOOR`. Values below this
# floor on the base side are treated as "effectively zero" and skipped
# for ratio computation — both because the ratio explodes and because
# the absolute drift is already gated by the runtime diff.
ABSOLUTE_FLOOR = 1e-12


def parse_tests(raw):
    """Extract the `tests` object from a baselines.json blob."""
    doc = json.loads(raw)
    return doc.get("tests", {})


def metrics_from_entry(entry):
    """Flatten one test entry into `{metric_label: value}` pairs.

    Vec3 metrics expand to per-component labels (`position_m[0]`,
    `position_m[1]`, `position_m[2]`). Scalar metrics use the bare
    key. Extras flatten by their `name` field.

    Returning a flat dict makes the diff loop a single zip without
    needing to track metric shape per pair.
    """
    out = {}
    for key, val in entry.items():
        if key == "extras":
            for extra in val or []:
                name = extra.get("name")
                value = extra.get("value")
                if name is not None and isinstance(value, (int, float)):
                    out[name] = float(value)
        elif isinstance(val, list) and len(val) == 3 and all(
            isinstance(x, (int, float)) for x in val
        ):
            out[f"{key}[0]"] = float(val[0])
            out[f"{key}[1]"] = float(val[1])
            out[f"{key}[2]"] = float(val[2])
        elif isinstance(val, (int, float)):
            out[key] = float(val)
    return out


try:
    base_tests = parse_tests(base_raw)
    current_tests = parse_tests(current_raw)
except json.JSONDecodeError as e:
    print(
        f"FAIL: cannot parse {baseline_path} as JSON: {e}",
        file=sys.stderr,
    )
    sys.exit(2)

warnings = []  # (test, metric, base_val, cur_val, ratio)
errors = []
tightenings = []  # informational only
new_tests = []
removed_tests = []

for test, base_entry in base_tests.items():
    if test not in current_tests:
        removed_tests.append(test)
        continue
    base_metrics = metrics_from_entry(base_entry)
    cur_metrics = metrics_from_entry(current_tests[test])
    for metric, base_val in base_metrics.items():
        if metric not in cur_metrics:
            # Metric removed — flag as a widening-class change so
            # silent dropouts of an asserted channel surface here
            # instead of slipping through.
            errors.append((test, metric, base_val, float("nan"), float("inf"),
                           "metric removed from current baseline"))
            continue
        cur_val = cur_metrics[metric]
        # Skip if the base-side value is effectively zero — see header.
        if abs(base_val) < ABSOLUTE_FLOOR:
            continue
        # Tightening (or no change): informational only.
        if cur_val <= base_val:
            if cur_val < base_val:
                tightenings.append((test, metric, base_val, cur_val))
            continue
        ratio = cur_val / base_val
        if ratio > FAIL_RATIO:
            errors.append((test, metric, base_val, cur_val, ratio, ""))
        elif ratio > WARN_RATIO:
            warnings.append((test, metric, base_val, cur_val, ratio))

for test in current_tests:
    if test not in base_tests:
        new_tests.append(test)


def fmt_pct(ratio):
    return f"{(ratio - 1.0) * 100.0:+.1f}%"


# Emit informational lines first (notices), then warnings, then
# errors. GitHub Actions renders all three in the job summary.
for test, metric, base_val, cur_val in tightenings:
    print(
        f"::notice::baseline tightened — {test}/{metric}: "
        f"{base_val:.3e} -> {cur_val:.3e} ({fmt_pct(cur_val / base_val)})"
    )

if new_tests:
    print(f"::notice::{len(new_tests)} new test(s) added to baselines.json "
          f"(no main-side value to compare):")
    for t in new_tests:
        print(f"::notice::  + {t}")

if removed_tests:
    print(f"::notice::{len(removed_tests)} test(s) removed from baselines.json "
          f"(present on {base_ref}, absent in working tree):")
    for t in removed_tests:
        print(f"::notice::  - {t}")

for test, metric, base_val, cur_val, ratio in warnings:
    print(
        f"::warning file={baseline_path}::baseline widened {fmt_pct(ratio)} "
        f"({ratio:.2f}x) for {test}/{metric}: "
        f"{base_val:.3e} -> {cur_val:.3e}. "
        f"Widening between {WARN_RATIO:.2f}x and {FAIL_RATIO:.2f}x triggers a warning; "
        f"document the physical justification in the PR description."
    )

for entry in errors:
    test, metric, base_val, cur_val, ratio, note = entry
    if note:
        msg = (
            f"::error file={baseline_path}::{note} — {test}/{metric}: "
            f"baseline was {base_val:.3e}"
        )
    else:
        msg = (
            f"::error file={baseline_path}::baseline widened {fmt_pct(ratio)} "
            f"({ratio:.2f}x) for {test}/{metric}: "
            f"{base_val:.3e} -> {cur_val:.3e}. "
            f"Widening past {FAIL_RATIO:.2f}x is a hard fail; refreeze deliberately "
            f"(see crates/astrodyn_bevy/tests/README.md 'Baseline-freeze workflow') "
            f"or back out the regression."
        )
    print(msg)

# Concluding human-readable summary line. GitHub picks up the annotations
# above regardless; the summary helps when reading raw logs locally.
n_w = len(warnings)
n_e = len(errors)
n_t = len(tightenings)
n_n = len(new_tests)
n_r = len(removed_tests)
print(
    f"baseline-widening: {n_e} error(s), {n_w} warning(s), "
    f"{n_t} tightening(s), {n_n} new, {n_r} removed "
    f"(thresholds: warn>{WARN_RATIO:.2f}x, fail>{FAIL_RATIO:.2f}x, base={base_ref})"
)

sys.exit(1 if n_e > 0 else 0)
PYEOF
