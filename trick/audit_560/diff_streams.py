#!/usr/bin/env python3
"""#560 operation-level FP-parity diff tool.

Parses two `[#560/FULL] step=N stage=K body=B op=<name> kI=vI ...` streams
(one from instrumented JEOD via `run_audit.sh`, one from our Rust harness
via `ASTRODYN_560_FULL_DUMP=1 cargo nextest run`) and reports the FIRST
operation per (op, body) where the streams disagree.

Alignment scheme:
    For each unique (op, body) pair, pair the Nth occurrence in JEOD with
    the Nth occurrence in ours. This is robust against JEOD and ours
    having different absolute step/stage indexing (e.g., JEOD may count
    pre-contact stages, ours may not).

Usage:
    python3 diff_streams.py \\
        --jeod <jeod_stderr.log> \\
        --ours <ours_stderr.log> \\
        [--max-occurrences N] [--threshold EPS]

Output:
    1. Per-(op, body) stream length comparison (catches op-count mismatches).
    2. First divergent occurrence per (op, body), sorted by first stream
       position.
    3. Top-20 largest |diff| at-occurrence-1 (catches state-level divergences
       early in the trajectory).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"\[#560/FULL\] "
    r"step=(?P<step>\d+) "
    r"stage=(?P<stage>\d+) "
    r"body=(?P<body>\d+) "
    r"op=(?P<op>\S+) "
    r"(?P<fields>.*)"
)
KV_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+(?:[eE][+-]?\d+)?)?)")


def parse_stream(path: Path, skip_all_zero: bool = True):
    """Return (per_op_body: dict, per_op_body_meta: dict).

    `per_op_body[(op, body)]` is a list of `{field: value}` dicts, ordered
    by occurrence in the stream.
    `per_op_body_meta[(op, body)]` parallel list of `(step, stage,
    line_index)` tuples for context when reporting divergences.

    `skip_all_zero` filters out records where every component is exactly
    zero — these are typically JEOD init-time calls before sim state is
    populated and would pollute the alignment.
    """
    records = defaultdict(list)
    meta = defaultdict(list)
    with path.open("r") as f:
        for line_idx, line in enumerate(f, 1):
            m = LINE_RE.search(line)
            if not m:
                continue
            step = int(m["step"])
            stage = int(m["stage"])
            body = int(m["body"])
            op = m["op"]
            fields = {k: float(v) for k, v in KV_RE.findall(m["fields"])}
            if skip_all_zero and fields and all(v == 0.0 for v in fields.values()):
                continue
            key = (op, body)
            records[key].append(fields)
            meta[key].append((step, stage, line_idx))
    return records, meta


def field_diff_norm(j: dict, o: dict) -> float:
    """L2 norm of per-field diff between two dicts (matched by key)."""
    s = 0.0
    for f in set(j.keys()) | set(o.keys()):
        d = j.get(f, 0.0) - o.get(f, 0.0)
        s += d * d
    return s ** 0.5


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jeod", type=Path, required=True)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--max-occurrences", type=int, default=200)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Only report |diff| above this (default 0).",
    )
    args = p.parse_args()

    jeod, jmeta = parse_stream(args.jeod)
    ours, ometa = parse_stream(args.ours)

    print(f"[diff] JEOD ops: {len(jeod)}; Ours ops: {len(ours)}")
    print(f"[diff] JEOD total dumps: {sum(len(v) for v in jeod.values())}")
    print(f"[diff] Ours total dumps: {sum(len(v) for v in ours.values())}")

    common = sorted(set(jeod.keys()) & set(ours.keys()))
    only_j = sorted(set(jeod.keys()) - set(ours.keys()))
    only_o = sorted(set(ours.keys()) - set(jeod.keys()))
    if only_j:
        print(f"[diff] Op-keys ONLY IN JEOD: {only_j}")
    if only_o:
        print(f"[diff] Op-keys ONLY IN OURS: {only_o}")

    # Per-(op, body) length comparison
    print()
    print("### PER-OP STREAM LENGTH COMPARISON")
    print(f"  {'op':<30} {'body':>4} {'#jeod':>7} {'#ours':>7} {'delta':>7}")
    length_mismatch = []
    for key in common:
        op, body = key
        nj = len(jeod[key])
        no = len(ours[key])
        if nj != no:
            length_mismatch.append((op, body, nj, no))
        if nj != no or len(common) < 30:
            mark = "  *" if nj != no else "   "
            print(f"  {op:<30} {body:>4} {nj:>7} {no:>7} {nj - no:>7}{mark}")
    if length_mismatch:
        print(f"  [WARN] {len(length_mismatch)} ops have different occurrence counts — alignment may slip.")

    # First divergent occurrence per (op, body)
    print()
    print("### FIRST DIVERGENT OCCURRENCE PER (op, body)")
    print(f"  {'op':<30} {'body':>4} {'occ':>5} {'step':>5} {'stage':>5} {'|diff|':>14}")
    first_div = []
    for key in common:
        op, body = key
        j_list = jeod[key]
        o_list = ours[key]
        n = min(len(j_list), len(o_list), args.max_occurrences)
        for i in range(n):
            d = field_diff_norm(j_list[i], o_list[i])
            if d > args.threshold:
                first_div.append((op, body, i, jmeta[key][i], j_list[i], o_list[i], d))
                break
    # Sort by first-divergent JEOD line position (so earliest-in-stream first)
    first_div.sort(key=lambda r: r[3][2])
    for op, body, occ, (step, stage, line), j, o, d in first_div[:30]:
        print(f"  {op:<30} {body:>4} {occ:>5} {step:>5} {stage:>5} {d:>14.6e}")

    # Detail of the very first divergence
    if first_div:
        op, body, occ, (step, stage, line), j, o, d = first_div[0]
        print()
        print("### FIRST DIVERGENCE (detail)")
        print(f"  op={op} body={body} occurrence={occ} step={step} stage={stage}")
        print(f"  |diff|_norm = {d:.6e}")
        for f in sorted(set(j.keys()) | set(o.keys())):
            jv = j.get(f, float("nan"))
            ov = o.get(f, float("nan"))
            dv = jv - ov
            tag = "  " if dv == 0.0 else " *"
            print(f"   {tag} {f}: jeod={jv:.17e} ours={ov:.17e} diff={dv:.6e}")
    else:
        print()
        print(f"[diff] NO divergences above threshold {args.threshold:.3e}.")
        print(f"[diff] Streams are bit-identical for all (op, body) occurrences.")
        return 0

    # Top divergences
    print()
    print("### TOP 20 |diff| AT OCCURRENCE 1 (catches earliest-state divergences)")
    occ1 = []
    for key in common:
        if not jeod[key] or not ours[key]:
            continue
        d = field_diff_norm(jeod[key][0], ours[key][0])
        occ1.append((key[0], key[1], d, jeod[key][0], ours[key][0]))
    occ1.sort(key=lambda r: r[2], reverse=True)
    print(f"  {'op':<30} {'body':>4} {'|diff|':>14}")
    for op, body, d, _j, _o in occ1[:20]:
        print(f"  {op:<30} {body:>4} {d:>14.6e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
