# astrodyn_verif_jeod_fixtures

Pure parsers and fixture loaders for JEOD reference data — no `astrodyn`
pipeline dependency.

This is a workspace-internal verification crate (`publish = false`). It sits
at the leaf level of the workspace alongside `astrodyn_quantities`, so any
owner crate (`astrodyn_math`, `astrodyn_dynamics`, `astrodyn_atmosphere`, …)
can pull it in as a dev-dep without dragging the full physics tree into its
test build. It owns:

- **JEOD-source parsers** (`apollo_truth`, `body_init_fixtures`, `mass_data`,
  `time_config`, `gravity_control`, `lvlh_init_data`, `orbital_init`,
  `reference_state`, `s_define`, …) that turn `Modified_data/*.py` and
  `S_define` into typed Rust values.
- **Trick CSV loaders** (`tier3_csv`, `dyncomp_csv`, `apollo_mass_tree`) that
  consume `log_state_ASCII.csv` outputs from `verif/SIM_*` Trick runs.
- **`crossval` `CrossvalReport`** used by Tier 3 trajectory tests to compute
  and assert per-component max errors against JEOD CSVs.

The simulation-driving rigs (`run_verification`, `verification`,
`tier3_report`, `tier3_baseline_diff`) live in `astrodyn_verif_jeod`, which
re-exports everything from this crate so existing
`astrodyn_verif_jeod::<module>::…` imports keep working from the upper-tree
consumers (`astrodyn_verif_parity`, the binaries).

Reference data still lives next door at
`crates/astrodyn_verif_jeod/test_data/` and `…/assets/`; the path resolver
in [`tier3_csv::test_data_path`] walks up to the workspace root and points
at that fixed location, so both crates resolve the same files.

For workspace context see the [Strategy](https://github.com/simnaut/astrodyn/wiki/Strategy)
and [Tier3-Regeneration](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration)
wiki pages.
