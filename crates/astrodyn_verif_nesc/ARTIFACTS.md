# NASA NESC Lunar Check-Case Artifact Index

Concise pointer list for the upstream artifacts this crate cross-validates
against. Primary identifier: **NESC-RP-23-01853**, *"Expansion of Check-Cases
for 6DOF Simulation"* (Koehler, Hawkins, Neuhaus et al., 2024-10-01) —
companion to the 2015 Earth-based predecessor NESC-RP-12-00770 /
NASA-TM-2015-218675.

## Final report

- [Volume I (main report) — NTRS 20240013031](https://ntrs.nasa.gov/citations/20240013031)
  ([PDF](https://ntrs.nasa.gov/api/citations/20240013031/downloads/20240013031.pdf))
- [Volume II Part 1 (Appendix A: plots) — NTRS 20240013556](https://ntrs.nasa.gov/citations/20240013556)
- [NESC Technical Bulletin 24-04 (overview) — NTRS 20240013467](https://ntrs.nasa.gov/citations/20240013467)
  ([PDF mirror on NESC Academy](https://nescacademy.nasa.gov/workshop/FlightSim/downloads/TB_24-04_103024a.pdf))
- [AAS conference paper (2024) — NTRS 20240016258](https://ntrs.nasa.gov/api/citations/20240016258/downloads/AAS_ExpandedCheckCases121024.pdf)

2015 Earth-based predecessor (NESC-RP-12-00770, NASA-TM-2015-218675):

- [Volume I summary PDF](https://nescacademy.nasa.gov/src/flightsim/Reports/NASA-TM-2015-218675-EOM_checkcase_summary.pdf)
- [Appendices PDF](https://nescacademy.nasa.gov/src/flightsim/Reports/NASA-TM-2015-218675-EOM_checkcase_appendices.pdf)

## Landing pages

- [NESC Academy flight-sim portal](https://nescacademy.nasa.gov/flightsim)
- [NASA 6DOF check-cases overview](https://www.nasa.gov/centers-and-facilities/nesc/6dof-check-cases/)
- [2015 (Earth-based) check-case index](https://nescacademy.nasa.gov/flightsim/2015)

## Common specifications (apply to every 2023 lunar case)

- [Body models (Apollo, etc.)](https://nescacademy.nasa.gov/flightsim/2023/bodies)
  — Apollo model anchor: [`#apollo-model`](https://nescacademy.nasa.gov/flightsim/2023/bodies#apollo-model)
- [Output specification (CSV channels, units, sign convention)](https://nescacademy.nasa.gov/flightsim/2023/output_specification)

## Lunar check cases (CC01–CC09b)

Each case page exposes a "Latest Results" table with per-participant CSVs
(`Lunar_<NN>_sim_<NN>.csv`) and, for reference-backed cases, a
`Lunar_<NN>_ref_01.csv`. The case pages render the download links via
JavaScript — browse to the page itself rather than `wget`-ing.

| Case  | Title                                       | Page                                                            |
| ----- | ------------------------------------------- | --------------------------------------------------------------- |
| CC01  | Keplerian propagation, Ref 1 ICs            | <https://nescacademy.nasa.gov/flightsim/2023/cc01>              |
| CC02  | Low-fidelity 8×8 GRAIL, Ref 1 ICs           | <https://nescacademy.nasa.gov/flightsim/2023/cc02>              |
| CC03  | High-fidelity 320×320 GRAIL, Ref 1 ICs      | <https://nescacademy.nasa.gov/flightsim/2023/cc03>              |
| CC04  | High circular orbit                         | <https://nescacademy.nasa.gov/flightsim/2023/cc04>              |
| CC05  | High circular orbit + perturbations         | <https://nescacademy.nasa.gov/flightsim/2023/cc05>              |
| CC05a | + tumbling                                  | <https://nescacademy.nasa.gov/flightsim/2023/cc05a>             |
| CC06  | Highly elliptical orbit                     | <https://nescacademy.nasa.gov/flightsim/2023/cc06>              |
| CC06a | HEO, no inertial rotation                   | <https://nescacademy.nasa.gov/flightsim/2023/cc06a>             |
| CC07  | HEO, Apollo body                            | <https://nescacademy.nasa.gov/flightsim/2023/cc07>              |
| CC08  | NRHO (Apollo body, 7-day, 60 s cadence)     | <https://nescacademy.nasa.gov/flightsim/2023/cc08>              |
| CC08a | NRHO, true anomaly 180°                     | <https://nescacademy.nasa.gov/flightsim/2023/cc08a>             |
| CC08b | NRHO, true anomaly 0°                       | <https://nescacademy.nasa.gov/flightsim/2023/cc08b>             |
| CC08c | NRHO, ΔR perturbation                       | <https://nescacademy.nasa.gov/flightsim/2023/cc08c>             |
| CC08d | NRHO, ΔV perturbation                       | <https://nescacademy.nasa.gov/flightsim/2023/cc08d>             |
| CC09  | Polar orbit, sensor position A              | <https://nescacademy.nasa.gov/flightsim/2023/cc09>              |
| CC09a | Polar orbit, sensor position B              | <https://nescacademy.nasa.gov/flightsim/2023/cc09a>             |
| CC09b | + moment profile                            | <https://nescacademy.nasa.gov/flightsim/2023/cc09b>             |

## Direct CSV directory (verified for CC08)

The `extract_nesc` binary fetches the eight CC08 participant CSVs from:

```
https://nescacademy.nasa.gov/flightsim/2023/scn_8/Lunar_08_sim_<NN>.csv   # NN = 01..08
```

The same `scn_<N>/Lunar_<NN>_sim_<NN>.csv` pattern is presumed to hold for
the other cases but has not been verified — confirm via the per-case page
before pinning regen URLs.

## Ephemeris dependencies

CC08 (and presumably the other modern-epoch lunar cases) specifies DE440 +
Moon principal-axes orientation. NAIF sources of the kernels we ship:

- DE440 short subset (1849–2150): [`de440s.bsp`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp)
- Moon PA (currently DE421-shared, 1900–2050): [`moon_pa_de421_1900-2050.bpc`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc)

## See also

- `crates/astrodyn_verif_nesc/README.md` — consensus methodology, status,
  regen workflow.
- `crates/astrodyn_verif_nesc/src/bin/extract_nesc.rs` — fetches CC08 CSVs
  from the verified `scn_8/` directory.
