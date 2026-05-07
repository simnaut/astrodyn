# astrodyn_verif_jeod

JEOD cross-validation parsers, fixtures, scenario rigs, and Tier 3 trajectory tests.

This is a workspace-internal verification crate (`publish = false`). It owns:

- **JEOD-source parsers** (`apollo_truth`, `body_init_fixtures`, `mass_data`,
  `time_config`, `gravity_control`, `lvlh_init_data`, `orbital_init`,
  `reference_state`, `s_define`, …) that turn `Modified_data/*.py` and
  `S_define` into typed Rust values.
- **Trick CSV loaders** (`tier3_csv`, `dyncomp_csv`, `apollo_mass_tree`) that
  consume `log_state_ASCII.csv` outputs from `verif/SIM_*` Trick runs.
- **Tier 3 verification scaffolding** (`crossval`, `verification`,
  `run_verification`) — the `VerificationCase` / `Tolerances` /
  `CsvReference` types, the per-scenario `sim_*` builders, and the
  `VerificationCaseExt::run_and_assert` driver.
- **Committed fixtures** under `test_data/` (180 reference trajectory CSVs,
  24 Apollo state snapshots, `baselines.{json,md}`, the verbatim NASA JEOD
  source mirror under `jeod_inputs/`, Tier 2 reference data under
  `body_init/` and `jeod_validation/`).
- **Tooling binaries** under `src/bin/`: `tier3_report`,
  `tier3_baseline_diff`, `extract_body_init`, `extract_jeod_validation`.

The 70 `tier3_*` integration tests under `tests/` drive the full
`astrodyn_runner::Simulation` pipeline end-to-end and assert against the
committed JEOD reference CSVs.

For workspace context see the [Strategy](https://github.com/simnaut/astrodyn/wiki/Strategy)
and [Tier3-Regeneration](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration)
wiki pages.
