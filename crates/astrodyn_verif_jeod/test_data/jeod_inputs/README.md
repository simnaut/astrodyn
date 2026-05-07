# `crates/astrodyn_verif_jeod/crates/astrodyn_verif_jeod/test_data/jeod_inputs/` — verbatim JEOD source mirror

Tier 3 verification rigs need JEOD configuration files (S_define
`#define DYNAMICS` lines, `Modified_data/*.py`, `SET_test/RUN_*/input.py`)
to recover dt, mass properties, time-scale offsets, and gravity-control
parameters. Pre #249 those were read live from `$JEOD_HOME`; now the
same files are committed verbatim under this directory so
`cargo nextest run --workspace` works on a fresh clone with no JEOD
checkout.

## Layout

Every fixture is committed at the same relative path it occupies in a
JEOD checkout. For example, `<jeod>/verif/SIM_dyncomp/Modified_data/mass.py`
is committed at
`crates/astrodyn_verif_jeod/test_data/jeod_inputs/verif/SIM_dyncomp/Modified_data/mass.py`. The
mirror is identical to JEOD's tree, so `diff -r ../jeod/<rel>
crates/astrodyn_verif_jeod/test_data/jeod_inputs/<rel>` confirms an unmodified copy.

Resolve a path from Rust via `astrodyn_test_data::jeod_inputs::path("<rel>")`.

## Refresh after a JEOD upgrade

```bash
# From the astrodyn workspace root, with $JEOD_HOME pointing at a
# JEOD checkout that already contains the upgraded source.
JEOD=$JEOD_HOME
DEST=test_data/jeod_inputs
find "$DEST" -type f ! -name README.md \
  | while read -r f; do
      rel=${f#"$DEST/"}
      cp "$JEOD/$rel" "$f"
    done
```

Re-run `cargo nextest run --workspace` and commit the diff.

## Add a new fixture

```bash
JEOD=$JEOD_HOME
REL=path/relative/to/jeod/root  # e.g. verif/SIM_xyz/S_define
mkdir -p "crates/astrodyn_verif_jeod/test_data/jeod_inputs/$(dirname "$REL")"
cp "$JEOD/$REL" "crates/astrodyn_verif_jeod/test_data/jeod_inputs/$REL"
```

Reference it from a Tier 3 rig with
`astrodyn_test_data::jeod_inputs::path("path/relative/to/jeod/root")`.

## Currently committed inputs

- `verif/SIM_dyncomp/` — `S_define`, `Modified_data/{mass,time,grav_controls,state}.py`, `SET_test/RUN_{3A,3B,7A,7B,7C,7D}/input.py` (consumed by `sim_dyncomp`, `sim_torque_simple`, `sim_tide_verif`, `sim_polar_motion`).
- `models/dynamics/derived_state/verif/Modified_data/date_and_time.py` (consumed by `sim_derived_state` SIM_NED rig).
- `models/dynamics/derived_state/verif/SIM_{OrbElem,LVLH,NED,Euler,Planetary,SolarBeta}/S_define`.
- `models/dynamics/derived_state/verif/SIM_SolarBeta/Modified_data/date_and_time.py`.
- `models/interactions/radiation_pressure/verif/SIM_3_ORBIT{,_1st_ORDER}/{S_define,Modified_data/date_and_time.py}` (consumed by `sim_srp`).
