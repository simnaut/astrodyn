#!/usr/bin/env python3
"""Bidirectional stream alignment + diff for issue #560.

Diagnostic-only. Consumes two `[#560/FULL] ...` streams (one from
JEOD, one from `ASTRODYN_560_FULL_DUMP=1 cargo nextest run ...`),
aligns them by `(op, body, occurrence-index)`, and reports the first
divergent op + a ranked table of all divergent ops.

This was the diff tool that produced the audit conclusion in
https://github.com/simnaut/astrodyn/issues/560: when both sides are
gated to only fire inside contact, the streams line up
position-by-position. Every input to the force kernel (`rel_pos`,
`geom_normal`, `geom_penetration_depth`, `rel_vel`, `v_normal_mag`)
matches bit-for-bit to 17 sig figs; the output `force_penetration_vec`
differs by exactly 1 ULP (1.7e-16 m). That ULP, multiplied by
`stiffness * dt` per stage and the ~1.2 per-stage amplification of
the stiff RK4, accumulates to the 2.5 mm mm-scale residual after the
152-stage contact event.

Run:

    python3 diff_streams.py --jeod jeod_dump.txt --rust rust_dump.txt

Outputs to stdout:

  1. The first line where alignment fails (op or occurrence-index
     mismatch).
  2. The first row with |delta| > 0.
  3. A ranked table of `(op, max_abs_delta, max_rel_delta)` for every
     op that ever differs.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

LINE_RE = re.compile(
    r"^\[#560/FULL\]\s+step=(?P<step>\d+)\s+stage=(?P<stage>\d+)\s+"
    r"body=(?P<body>\d+)\s+op=(?P<op>\S+)\s+(?P<fields>.+)$"
)
FIELD_RE = re.compile(r"(?P<key>\S+?)=(?P<value>-?\d+\.\d+e[+-]?\d+|-?\d+(?:\.\d+)?)")


@dataclass
class Entry:
    """One parsed `[#560/FULL] ...` line."""

    step: int
    stage: int
    body: int
    op: str
    fields: dict[str, float]
    raw: str


def parse(path: str) -> list[Entry]:
    """Parse every `[#560/FULL]` line from `path`. Lines without the
    prefix are silently dropped (the captured stream may carry other
    test / build chatter mixed in)."""

    out: list[Entry] = []
    with open(path, "r") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            m = LINE_RE.match(raw)
            if not m:
                continue
            fields: dict[str, float] = {}
            for fm in FIELD_RE.finditer(m.group("fields")):
                try:
                    fields[fm.group("key")] = float(fm.group("value"))
                except ValueError:
                    # Skip non-numeric fields. Should never trigger
                    # given the format helpers emit `{:.17e}`, but
                    # defends against accidental contamination.
                    continue
            if not fields:
                continue
            out.append(
                Entry(
                    step=int(m.group("step")),
                    stage=int(m.group("stage")),
                    body=int(m.group("body")),
                    op=m.group("op"),
                    fields=fields,
                    raw=raw,
                )
            )
    return out


@dataclass
class Divergence:
    """Per-op aggregated divergence statistics over the full stream."""

    op: str
    count: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    first_step: int = -1
    first_stage: int = -1
    first_body: int = -1
    first_key: str = ""
    first_jeod: float = 0.0
    first_rust: float = 0.0
    keys: set[str] = field(default_factory=set)


def aligned_pairs(jeod: list[Entry], rust: list[Entry]) -> list[tuple[Entry, Entry]]:
    """Align by `(op, body, occurrence-index)`.

    We track, per `(op, body)`, the next occurrence on each side. The
    first occurrence on each side is paired up, then the second, and so
    on. This matches the alignment the JEOD-side patch and Rust-side
    dump were designed for: both sides emit a deterministic, ordered
    sequence of ops per stage, so positional alignment within each
    `(op, body)` bucket is well-defined.
    """

    from collections import defaultdict

    jeod_by_key: dict[tuple[str, int], list[Entry]] = defaultdict(list)
    rust_by_key: dict[tuple[str, int], list[Entry]] = defaultdict(list)
    for e in jeod:
        jeod_by_key[(e.op, e.body)].append(e)
    for e in rust:
        rust_by_key[(e.op, e.body)].append(e)

    pairs: list[tuple[Entry, Entry]] = []
    for key, j_list in jeod_by_key.items():
        r_list = rust_by_key.get(key, [])
        n = min(len(j_list), len(r_list))
        for i in range(n):
            pairs.append((j_list[i], r_list[i]))
        if len(j_list) != len(r_list):
            print(
                f"warning: op={key[0]} body={key[1]} occurrence count "
                f"jeod={len(j_list)} rust={len(r_list)} — truncating to {n}",
                file=sys.stderr,
            )
    return pairs


def diff(jeod_path: str, rust_path: str) -> int:
    jeod = parse(jeod_path)
    rust = parse(rust_path)
    if not jeod:
        print(f"error: no [#560/FULL] lines found in {jeod_path}", file=sys.stderr)
        return 2
    if not rust:
        print(f"error: no [#560/FULL] lines found in {rust_path}", file=sys.stderr)
        return 2

    print(f"jeod: parsed {len(jeod)} entries from {jeod_path}")
    print(f"rust: parsed {len(rust)} entries from {rust_path}")

    pairs = aligned_pairs(jeod, rust)
    print(f"aligned: {len(pairs)} (op, body, occurrence) pairs")

    by_op: dict[str, Divergence] = {}
    first_divergent: tuple[Entry, Entry, str, float, float] | None = None
    for j, r in pairs:
        for key, j_val in j.fields.items():
            r_val = r.fields.get(key)
            if r_val is None:
                continue
            delta = abs(j_val - r_val)
            if delta == 0.0:
                continue
            denom = max(abs(j_val), abs(r_val), 1e-300)
            rel = delta / denom
            d = by_op.setdefault(j.op, Divergence(op=j.op))
            d.count += 1
            d.keys.add(key)
            if delta > d.max_abs:
                d.max_abs = delta
            if rel > d.max_rel:
                d.max_rel = rel
            if d.first_step == -1:
                d.first_step = j.step
                d.first_stage = j.stage
                d.first_body = j.body
                d.first_key = key
                d.first_jeod = j_val
                d.first_rust = r_val
            if first_divergent is None:
                first_divergent = (j, r, key, j_val, r_val)

    if first_divergent is None:
        print("\nresult: streams agree to 17 sig figs on every aligned op/key.")
        return 0

    j, r, key, j_val, r_val = first_divergent
    delta = abs(j_val - r_val)
    print("\nfirst divergent line:")
    print(f"  jeod: {j.raw}")
    print(f"  rust: {r.raw}")
    print(
        f"  Δ({key}) = {delta:.3e} "
        f"(jeod={j_val:.17e} rust={r_val:.17e}, "
        f"step={j.step} stage={j.stage} body={j.body} op={j.op})"
    )

    print("\nranked table (max |Δ| per op):")
    ranked = sorted(by_op.values(), key=lambda d: d.max_abs, reverse=True)
    width_op = max(len(d.op) for d in ranked) if ranked else 8
    print(f"  {'op':<{width_op}}  {'count':>6}  {'max_abs':>12}  {'max_rel':>12}")
    for d in ranked:
        print(
            f"  {d.op:<{width_op}}  {d.count:>6}  "
            f"{d.max_abs:>12.3e}  {d.max_rel:>12.3e}"
        )

    # Diagnostic-only — never fail the CI on divergence count. Exit 1
    # signals "streams differ", which is the audit's *expected* output
    # whenever it runs (the audit conclusion is that the divergence is
    # intrinsic to two independent f64 implementations).
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jeod", required=True, help="path to JEOD-side dump file")
    ap.add_argument("--rust", required=True, help="path to Rust-side dump file")
    args = ap.parse_args()
    return diff(args.jeod, args.rust)


if __name__ == "__main__":
    sys.exit(main())
