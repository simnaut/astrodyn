# astrodyn_verif_nesc

Cross-validation track against the NASA NESC GN&C Lunar Check Cases
(NESC-RP-23-01853, "Expansion of Check-Cases for 6DOF Simulation"). This
crate is parallel to `astrodyn_verif_jeod`: each `tier3_nesc_*` test
propagates `astrodyn_runner::Simulation` from NESC-published initial
conditions and asserts position / velocity / attitude against the
case's reference trajectory at the published checkpoint cadence.

## Status

| Case | Status | Notes |
|------|--------|-------|
| CC8 (NRHO) — runner ↔ NESC | passing translation + attitude | sim_01 baseline; consensus methodology pending |
| CC8 (NRHO) — bevy ↔ runner   | passing (translation + attitude bit-identity) | unblocked by `populate_app::<P>` per-planet auto-registration fix |

The DE440 SPK and the NESC sim_01 reference CSV are committed, the
recipe layer (`epoch::at_iso`, `moon::grail150_with_libration`,
`SimulationBuilder::add_third_body_with_ephemeris`,
`recipes::vehicle::nesc_apollo_lm`, `recipes::ephemeris::de440_with_moon_pa`)
is in place, and the runner-side test runs end-to-end with **both
translational and attitude assertions live**. Translational fit to
sim_01 over 7 days is ~150 m magnitude; attitude fit is ~0.15 rad
(~8.4°) over the 605° body-frame spin. Per-component tolerances are
set at observed-max × 1.05 per the CLAUDE.md rule.

## Cases

### CC8 — Lunar Near-Rectilinear Halo Orbit

- Source: <https://nescacademy.nasa.gov/flightsim/2023/cc08>
- Body model: <https://nescacademy.nasa.gov/flightsim/2023/bodies#apollo-model>
- Force model: Moon central (8×8 GRAIL), Earth + Sun third bodies, DE440
  ephemeris. Gravity only — no SRP, no drag, no gravity-gradient torque.
- Vehicle: Apollo body (16 642 kg, full inertia tensor incl. off-diagonal
  terms, CoM offset (4.7, 0.01, −0.0075) m).
- Duration: 7 days at 60 s checkpoint cadence; 6-DoF (translation +
  attitude).

## Reference methodology — single sim today, consensus tomorrow

NESC checkcases come in two flavors:

- **Reference-backed** (cc01, cc01o, …): a `Lunar_<NN>_ref_01.csv` from
  an analytical / exact solution. Tests assert `|ours − ref| < tol`.
  Tolerance interpretation: how close are we to truth.
- **Consensus-only** (cc02..cc09b including CC8): only
  `Lunar_<NN>_sim_<NN>.csv` files from 8 participating organizations.
  No single ground truth — the published validation methodology is
  inter-simulation agreement.

CC8 is consensus-only. Today's test uses **sim_01** as the reference
by convention (six of eight sims agree at IC to ≥ 9 decimal places —
sim_03 and sim_06 are anomalies). This is a placeholder. The right
methodology is below.

### Why "fit-to-family" matters for cc08

The 7-day NRHO is genuinely sensitive: small differences in Earth
ephemeris time-tagging, RK4 step size, or 8×8 GRAIL coefficient
rounding accumulate at the apolune. A bare `|ours − sim_01| < tol`
test confuses two errors:

1. **Our trajectory's drift** from physics truth (what we want to bound).
2. **sim_01's drift** from physics truth (sim_01 has its own integrator
   noise; we end up matching its biases as well as its physics).

Without consensus statistics we can't distinguish "our bug" from "sim_01
is noisier than sim_05 in this regime" — both look like out-of-tolerance
asserts.

### Proposed methodology: z-score against median, MAD-based spread

For each consensus-only case, at each checkpoint `t` and channel `c`
(position component, velocity component, etc.), the regen binary
computes across the `N` in-family sims:

- **Median**: `μ̃(t, c) = median(sim_1[t][c], …, sim_N[t][c])`
- **Spread**: `σ̃(t, c) = MAD(sim_1[t][c], …, sim_N[t][c])` —
  median absolute deviation, robust to a single divergent participant.

The committed reference is then a single `cc<NN>_<name>_consensus.csv`
with **two values per channel** (median + MAD), plus an
`cc<NN>_<name>_in_family.json` listing which `sim_NN` participants
were included after IC anomaly detection (a sim is excluded if its
t=0 row deviates from the median by > k_ic·MAD_ic on any channel).

The test computes per-channel:

- **Bias**: `|ours[t][c] − μ̃(t, c)|` — raw distance from median.
- **Z-score**: `bias / max(σ̃(t, c), σ_floor[c])` — normalized;
  z = 1 means "as far from median as the typical participant is."
  `σ_floor[c]` absorbs numerical noise where the inter-sim spread
  collapses to zero (e.g., near IC).
- **Envelope hit-rate**: fraction of timesteps where
  `ours[t][c] ∈ [μ̃ ± k·σ̃]`.

The verdict structure:

| max z-score | interpretation | test outcome |
|-------------|----------------|--------------|
| ≤ 1 | indistinguishable from a typical participant | PASS, green |
| 1 < z ≤ 3 | "in family" — within the spread | PASS |
| 3 < z ≤ 5 | tail of family — flagged | PASS with warning |
| > 5 | outlier — investigate | FAIL |

`σ_floor` per channel (initial values, will be tuned after first regen):

| Channel | Floor |
|---------|-------|
| position component | 1 m |
| velocity component | 1 mm/s |
| quaternion component | 1e-6 |
| body angular velocity component | 1e-9 rad/s |

### Why median + MAD over alternatives

- **Single-sim baseline** (today): simple, but matches one propagator's
  noise instead of the cluster's physics. We lose "are we in family?"
  signal.
- **Mean + std**: not robust to outliers — sim_03 / sim_06 anomalies
  pull the mean and inflate std, giving us a softer test that absorbs
  legitimate bugs.
- **Min/max envelope**: simple and intuitive, but no normalized "how
  good am I?" metric. A single divergent participant pushes the
  envelope wide for everyone.
- **RMS distance to consensus**: useful summary, but loses per-channel
  diagnostic value when only one component is misbehaving.

The median + MAD pair is the median-statistics analog of mean + std,
robust to outliers, and gives a per-channel z-score that scales with
the legitimate inter-sim disagreement. NRHO spread grows at apolune;
MAD grows with it; tolerances auto-track the physics.

### Implementation sketch (next iteration)

1. **`extract_nesc` extension**: instead of emitting just sim_01, fetch
   all 8 sim CSVs, run IC anomaly detection (drop sims whose t=0 row
   deviates > 10·MAD_ic from the in-family median), compute median +
   MAD per channel per checkpoint across the surviving sims, and
   emit `cc8_nrho_consensus.csv` (with both values per channel) + a
   small `cc8_nrho_in_family.json` documenting the inclusion list and
   the `σ_floor` constants.
2. **`astrodyn_verif_nesc::consensus` module**: a `ConsensusRef`
   loader that reads the consensus CSV and exposes
   `bias(timestep, channel)`, `z_score(timestep, channel)`, and
   `assert_in_family(threshold)` helpers parallel to
   `CrossvalReport::assert_*`.
3. **Test rewrite**: `tier3_nesc_cc8_nrho` switches from
   `report.assert_position(...)` to
   `consensus.assert_in_family(z_max = 3.0)`, with a fall-through
   warning at z > 3 and panic at z > 5.

Estimated work: ~250 LOC + the regen run. Asset size impact: the
consensus CSV is ~7 MB committed (14 channels × 2 stats × 10081 rows ×
~25 bytes), replaces the current 2.7 MB sim_01 CSV. The 8 raw sim CSVs
(~105 MB) stay non-committed; they're fetched by the regen binary
on demand.

## Regenerating reference data

The committed CSV under `test_data/` is produced by the regen binary
either fetching directly from the NESC site or from a locally-mirrored
NESC artifact tree.

```bash
# Default: fetch over HTTPS from nescacademy.nasa.gov.
cargo run -p astrodyn_verif_nesc --bin extract_nesc

# Or use a locally-staged mirror (the binary expects the NESC site's
# scn_8/ subdirectory layout under --nesc-home):
cargo run -p astrodyn_verif_nesc --bin extract_nesc -- \
    --nesc-home /path/to/nesc/release
```

The binary parses the upstream `Lunar_08_sim_<NN>.csv` files by
**header name** (robust against trailing extra columns some sims
emit), projects to our 14 canonical channels, converts angular rates
from deg/s to rad/s, and writes
`crates/astrodyn_verif_nesc/test_data/cc8_nrho_reference.csv` with
17-significant-digit f64 values for round-trip reproducibility.

The post-consensus iteration of the binary will additionally fetch
all eight sims, compute median + MAD, and emit the consensus CSV;
the test side then asserts `z_score < 3` instead of raw bias.

## DE440 ephemeris

CC8 specifies DE440. The crate ships:

- `crates/astrodyn_ephemeris/assets/de440.bsp` — the NAIF
  [`de440s.bsp`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp)
  short-subset (~32 MB, 1849–2150). Sufficient for the modern epochs
  we currently target; the full DE440 archive is two orders of
  magnitude larger.
- `crates/astrodyn_ephemeris/assets/moon_pa_de421_1900-2050.bpc` —
  Moon principal-axes orientation kernel (~1.7 MB), shared between
  DE421 and DE440 recipes.

The recipes `astrodyn::recipes::ephemeris::de440()` (SPK only) and
`de440_with_moon_pa()` (SPK + BPC libration) load these from the
embedded byte slices. Mixing a DE440 SPK with the DE421 BPC introduces
a sub-arcsecond inconsistency in Moon libration over a few-day
propagation; tightening that by adding a `moon_pa_de440_*.bpc` is a
follow-up.

## Open questions

- **Canonical NESC release pin** — the public artifact tree at
  `https://nescacademy.nasa.gov/flightsim/2023/scn_<N>/` is the
  current source. The tree is hot-updated as participants resubmit;
  pin a date-checkpointed snapshot once the consensus methodology
  lands so regen reproducibility is guaranteed.
- ~~**Attitude-integrator divergence**~~ — *resolved (#454).* NESC
  publishes IC and reference quaternions as right-transformative
  (NESC-RP-23-01853 §7.4.1); our `JeodQuat` is left-transformative.
  `cc8.rs` now conjugates the vector part at the IC boundary and the
  CSV parser. Residual attitude drift vs sim_01 over 7 days is
  ~0.15 rad (~8.4°), consistent with inter-propagator numerical noise.

## See also

- Issue #399 — verification track umbrella issue
- `crates/astrodyn_verif_jeod/` — parallel JEOD/Trick verification track
- `crates/astrodyn_verif_parity/tests/bevy_parity_nesc_*.rs` — Bevy ↔
  runner bit-identity wrappers
