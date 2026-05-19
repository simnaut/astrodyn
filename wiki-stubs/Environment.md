# Environment

`CLAUDE.md` notes that a fresh clone supports the full test suite with no
`$JEOD_HOME`. This page covers the regen workflows that *do* need external
checkouts and the optional verification track.

## Fresh-clone baseline

`cargo build --workspace && cargo nextest run --workspace` works on a fresh
clone of this repo with no JEOD checkout — every test (unit, Tier 2, **and
Tier 3**) reads from committed fixtures under `test_data/`. CI's "Test (unit +
tier 2)" and Tier 3 jobs each include a regression fence that asserts
`JEOD_HOME` and `JEOD_PATH` are both unset before running.

## When you need `$JEOD_HOME`

You only need `$JEOD_HOME` when **regenerating fixtures** after a JEOD
upgrade.

The `extract_*` regen binaries are distributed by owner crate:

- `extract_grav_coeffs` and `extract_mars_data` live in
  `crates/astrodyn_gravity/src/bin/` (parsing JEOD `.cc` files into
  `crates/astrodyn_gravity/test_data/gravity/*.bin`).
- `extract_planet_pfixposn` lives in `crates/astrodyn_planet/src/bin/`.
- `extract_eop_table` lives in `crates/astrodyn_time/src/bin/` (parsing JEOD's
  IERS EOP table into `crates/astrodyn_time/test_data/eop/iers_eop_c04.bin`).
- `extract_body_init` / `extract_jeod_validation` live in
  `crates/astrodyn_verif_jeod/src/bin/`.

The verbatim NASA JEOD source mirror under
`crates/astrodyn_verif_jeod/test_data/jeod_inputs/` is refreshed via the `cp`
recipe in that directory's `README.md`. All flows accept either `$JEOD_HOME`
or `--jeod-home <PATH>`.

## Tier 3 reference-CSV regen (Docker)

For the Docker reference-CSV regen flow (Tier 3 baselines):

```bash
cd /home/user/git   # or wherever your repos live
git clone https://github.com/nasa/jeod.git
git clone https://github.com/nasa/trick.git

export JEOD_HOME=$(pwd)/jeod
export TRICK_HOME=$(pwd)/trick

cargo xtask regenerate-tier3
```

`JEOD_HOME` is the standard NASA convention; the older `JEOD_PATH` alias was
retired in #239. `$TRICK_HOME` is required only by the Docker reference-CSV
regen flow. The full workflow (incremental vs `--force`, adding a new sim,
troubleshooting matrix, `log_state_ASCII.csv` column layout) lives on
[Tier3-Regeneration](Tier3-Regeneration).

## NESC GN&C Lunar Check Cases

The parallel **NESC GN&C Lunar Check Cases** verification track lives in
`crates/astrodyn_verif_nesc/`. Its regen binary is `extract_nesc`
(`crates/astrodyn_verif_nesc/src/bin/extract_nesc.rs`); it accepts
`$NESC_HOME` or `--nesc-home <PATH>` and writes parsed CSVs into
`crates/astrodyn_verif_nesc/test_data/`. See
`crates/astrodyn_verif_nesc/README.md` for the workflow, the canonical
release pin, and the DE440 ephemeris asset that CC8 depends on.

## Ephemeris kernels

`astrodyn_ephemeris` distributes its required DE4xx kernels as assets on the
project's GitHub Releases (introduced in #476). The default `fetch` feature
downloads them on first use; for air-gapped builds, set
`$ASTRODYN_EPHEMERIS_KERNELS_DIR` to a directory holding the pre-downloaded
kernels and disable the feature (`--no-default-features`).
