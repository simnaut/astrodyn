# jeod_test_data

JEOD reference-data parsers, Tier 2 fixtures, and Tier 3 baseline
tooling for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Parses JEOD's `Modified_data/*.py`, `S_define`, `Leap_Second.dat`, the
gravity `data/include/*.cc` coefficient files, the `verif_out.txt`
gravity-test vectors, and the Trick CSV reference logs. Sources are
under
[NASA JEOD v5.4.0](https://github.com/nasa/jeod/tree/jeod_v5.4.0).

## Layered architecture

```
bevy_jeod / mission code     (production)
   ↑    only for tests / dev / regen
jeod_test_data               ←  this crate (dev / test only)
   ↓
jeod_quantities, jeod_gravity, jeod_math
```

`jeod_test_data` is **not** part of the production code path. Its
parsers populate the binary fixtures committed under `test_data/`; the
production `jeod_*` crates read those committed fixtures and never
parse JEOD source themselves.

## Public surface

- `jeod_inputs` — committed mirror of JEOD `.py` / `.dat` source files.
- `jeod_cc` — JEOD `.cc` gravity-coefficient parser; produces
  `SphericalHarmonicsData`.
- `gravity_fixtures` — load committed binary coefficient fixtures.
- `gravity_verif` — `verif_out.txt` test-vector parser.
- `body_init_fixtures`, `mass_data`, `orbital_init`, `reference_state`,
  `time_config`, `s_define`, `leap_second`, `gravity_control` — JEOD
  Trick `Modified_data/*.py` parsers.
- `tier3_csv`, `dyncomp_csv`, `apollo_truth`, `apollo_mass_tree`,
  `crossval` — Tier 3 reference-CSV loaders and the cross-validation
  report.
- `extract_*` binaries (under `src/bin/`) regenerate fixtures from a
  fresh `$JEOD_HOME` checkout.

## See also

- [Project README](../../README.md) and
  [`CLAUDE.md`](../../CLAUDE.md) — Tier 1 / 2 / 3 conventions, fixture
  layout, regen workflow.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_test_data/>
