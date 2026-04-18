# Earth Lighting Validation State

**Status:** Closed with documented rationale. No further action feasible
(tracked in [#49](https://github.com/simnaut/bevy_jeod/issues/49)).

## The gap

No JEOD verification sim propagates a vehicle through shadow entry/exit with
`EarthLighting` logged alongside the integrated state. A full Tier 3
trajectory cross-validation — the kind where our `Simulation::step()` output
is compared against a JEOD reference trajectory at 60 s checkpoints — is
therefore not possible for earth lighting. The gap is in JEOD's verification
sim suite, not in our port: we have ported the lighting physics
(`jeod_sim::compute_earth_lighting`), but the reference data to cross-validate
a propagating test against does not exist upstream.

An exhaustive grep of every `S_define` in JEOD 5.4 finds exactly one sim that
includes `EarthLighting`: `SIM_LIGHT_CIR`. `SIM_dyncomp` and the other
propagating sims never instantiate the model.

## What SIM_LIGHT_CIR covers

`models/environment/earth_lighting/verif/SIM_LIGHT_CIR/` exercises
`circle_intersect()` and `calc_lighting()` at **static parametric geometry
inputs**, one per RUN directory (`RUN_T01_LIGHT_VER` … `RUN_T10_LIGHT_VER` =
10 cases). The sim uses a stub ephemeris (`LightingEphem`) — no real planet
positions, no orbit propagation, no `DynBody`, no integration loop. The CSV
outputs also suffer from a Trick DRAscii scheduling lag (each row's lighting
fields reflect the previous step's inputs), which makes even the static
outputs unusable for direct numerical comparison.

## What would be needed for a propagating validation

A JEOD sim that combines the `SIM_dyncomp` integration loop with an
`EarthLighting` sim object:

1. `DynBody` + integration loop (as in `SIM_dyncomp`) with Sun, Moon, and
   Earth registered on `DynManager`.
2. An `EarthLighting` instance initialized from that `DynManager`.
3. `calc_lighting(vehicle_position)` called each dynamics step.
4. Logged output pairing vehicle inertial position with `sun_earth`,
   `moon_earth`, and `earth_albedo` fields at each step so that eclipse entry
   and exit can be timed against our propagated trajectory.

Issue [#49](https://github.com/simnaut/bevy_jeod/issues/49) sketches a
concrete S_define addition. No such sim exists in JEOD 5.4 today.

## Our current coverage

Our 11 `tier3_bevy_earth_lighting_*` tests in
`tests/bevy_parity_lighting.rs` validate that the Bevy pipeline
(`EarthLightingPlugin` inside `Simulation::step()`) produces bit-identical
results to a direct call through `jeod_sim`. They do **not** cross-validate
against JEOD reference output — they establish Bevy-vs-`jeod_sim` parity on
the 10 `SIM_LIGHT_CIR` geometric cases plus an integrated ISS-orbit pipeline
case:

| Test | Label | What it checks |
|------|-------|----------------|
| `tier3_bevy_earth_lighting_t01` | `t01_sunlit` | Vehicle on sun side, clear lighting. |
| `tier3_bevy_earth_lighting_t02` | `t02_shadow` | Vehicle in Earth's umbra. |
| `tier3_bevy_earth_lighting_t03` | `t03_terminator` | Vehicle on day/night terminator. |
| `tier3_bevy_earth_lighting_t04` | `t04_moon_inline` | Moon between Sun and vehicle. |
| `tier3_bevy_earth_lighting_t05` | `t05_geo_sunlit` | GEO altitude, sunlit. |
| `tier3_bevy_earth_lighting_t06` | `t06_polar` | Polar position. |
| `tier3_bevy_earth_lighting_t07` | `t07_offset_sun_moon` | Sun/Moon off principal axes. |
| `tier3_bevy_earth_lighting_t08` | `t08_deep_shadow` | Vehicle deep inside Earth's umbra. |
| `tier3_bevy_earth_lighting_t09` | `t09_moon_near_veh_dir` | Moon near the vehicle line-of-sight. |
| `tier3_bevy_earth_lighting_t10` | `t10_coplanar_45deg` | Coplanar Sun/Moon/vehicle at 45°. |
| `tier3_bevy_earth_lighting_pipeline` | ISS LEO, `NUM_STEPS` | Full `Simulation::step()` pipeline with lighting computed each dynamics step. |

Assertions compare `sun_earth`, `moon_earth`, and `earth_albedo` field-by-field
at IEEE-754 bit equality (`assert_earth_lighting_eq` in
`tests/parity_helpers/mod.rs`). The 10 geometric cases mirror the
`SIM_LIGHT_CIR` RUN inputs; the pipeline test establishes that integration
through the Bevy scheduler does not disturb lighting output.

Additional lower-tier coverage: `tier3_sim_earth_lighting_consistency`
(determinism + physical bounds on sun occlusion/visible complement), and
unit tests of `circle_intersect` in `crates/jeod_interactions/src/earth_lighting.rs`.

## Conclusion

Closed with documented rationale; no further action feasible. Bridging the
gap requires a new JEOD verification sim that propagates a vehicle through
lighting transitions and logs the state at each step. If one appears
upstream (or we decide to author and submit one), a propagating Tier 3 test
can be added following the pattern of `crates/jeod_runner/tests/tier3_sim_shadow_2a.rs`.

## References

- JEOD model: `$JEOD_HOME/models/environment/earth_lighting/`
  (`src/earth_lighting.cc`, `include/earth_lighting.hh`)
- JEOD sim: `$JEOD_HOME/models/environment/earth_lighting/verif/SIM_LIGHT_CIR/`
  (S_define, `SET_test/RUN_T01_LIGHT_VER` … `RUN_T10_LIGHT_VER`)
- Our parity tests: `tests/bevy_parity_lighting.rs`
- Our lighting implementation: `crates/jeod_interactions/src/earth_lighting.rs`
  (re-exported through `jeod_sim::compute_earth_lighting`)
- Issue: [#49](https://github.com/simnaut/bevy_jeod/issues/49)
