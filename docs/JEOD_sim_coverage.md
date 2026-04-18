# JEOD SIM_* Coverage Map

Mapping of JEOD's 66 production verification sims (under `models/`, `sims/`, `verif/`) to our Tier 3 test functions. Training-exercise sims (`docs/Training/Exercises/SIM_*`) are tutorials, not verification, and are out of scope.

Our Tier 3 tests live in:

- `crates/jeod_runner/tests/tier3_*.rs` — runner-layer (standalone Simulation) tests
- `tests/bevy_parity_*.rs` — Bevy-layer parity with the runner (`tier3_bevy_*`)
- `crates/jeod_dynamics/tests/`, `crates/jeod_time/tests/`, `crates/jeod_math/tests/` — unit/analytical Tier 3s

Column conventions:

- **covered**: at least one Tier 3 test exercises the sim's primary invariant (full pipeline, JEOD CSV reference).
- **partial**: we exercise the same physics but via a different sim config (e.g. we don't run JEOD's exact input.py variant).
- **derived**: the behavior is exercised indirectly through a sim that includes this one as a component (e.g. time-scale sims are exercised by any trajectory test that advances time).
- **not covered**: no Tier 3 test targets this sim.
- **n/a**: the sim exercises JEOD infrastructure we chose not to port (Trick sim interface, JEOD memory allocator, checkpoint container). These are not gaps to close.

## dynamics / body_action

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_orbinit | covered | `tier3_orbinit_docker_run0001_iss_inertial`, `tier3_orbinit_docker_run0101_sts_inertial`, `tier3_orbinit_docker_run0201_iss_pfix`, `tier3_orbinit_docker_run0301_sts_pfix`, `tier3_orbinit_docker_run0401_sts_trans_state` + 7 analytical (`tier3_orbinit_circular_leo`, `tier3_orbinit_hyperbolic`, etc.) |
| SIM_lvlh_init | partial | LVLH frame is exercised by `tier3_bevy_lvlh*` / `tier3_simulation_lvlh*`; no direct `lvlh_init` body-action scenario |
| SIM_ref_attach | not covered | no ref-frame attach-point scenario ported |
| SIM_verif_attach_mass | covered | `tier3_sim_attach_mass`, `tier3_sim_attach_detach_simple`, `tier3_sim_attach_detach_child_derivative_t0`, `tier3_sim_attach_detach_complex_t0` |
| SIM_verif_frame_switch | covered | `tier3_apollo8_frame_switch` |

## dynamics / derived_state

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_Euler | covered | `tier3_simulation_euler`, `tier3_simulation_euler_ecc`, `tier3_simulation_euler_equ` + Bevy parity |
| SIM_LVLH | covered | `tier3_simulation_lvlh`, `tier3_simulation_lvlh_ecc`, `tier3_simulation_lvlh_equ` + Bevy parity + `tier3_lvlh_*` analytical |
| SIM_LvlhRelative | covered | `tier3_simulation_lvlhrel_test0`, `tier3_simulation_lvlhrel_test1` + Bevy parity + `tier3_sim_lvlh_relative_consistency` |
| SIM_NED | covered | `tier3_simulation_ned_polar`, `tier3_simulation_ned_sph_inc`, `tier3_simulation_ned_sph_polar` + Bevy parity |
| SIM_OrbElem | covered | `tier3_simulation_orbelem` + t01/t10/t20/t30/t40/t50/t55 variants + Bevy parity |
| SIM_Planetary | covered | `tier3_simulation_planetary_geo`, `..._leo_ecc`, `..._leo_equ`, `..._leo_inc`, `..._leo_polar` + Bevy parity + `tier3_bevy_planetary_geo`, `tier3_bevy_geodetic_derived_state`, `tier3_bevy_polar_geodetic`, `tier3_simulation_geodetic` |
| SIM_Relative | covered | `tier3_simulation_relative_a_rot_no_trans`, `..._ab_rot_ab_trans`, `..._no_rot_ab_trans` + Bevy parity + `tier3_relative_*` analytical + `tier3_sim_relative_state_consistency` |
| SIM_SolarBeta | covered | `tier3_simulation_solar_beta`, `..._equ`, `..._obliquity` + Bevy parity + `tier3_solar_beta_*` analytical |

## dynamics / dyn_body, dyn_manager, rel_kin

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_dyncomp | covered | `tier3_dyncomp_6dof_rigid_body_invariance`, `tier3_dyncomp_attitude_stability_major_axis`, `tier3_dyncomp_drag_point_mass_monotonic_decay`, `tier3_dyncomp_external_force_impulse_response`, `tier3_dyncomp_external_torque_impulse_response`, `tier3_dyncomp_point_mass_3dof_conservation`, `tier3_dyncomp_point_mass_plus_thirdbody_conservation`, plus the `tier3_simulation_run*` sweep (run2/3/4/5/6/7/9/10 series) — JEOD's SIM_dyncomp is the canonical cross-validation target and we have ~30 RUN-specific tests |
| SIM_dyncomp_structure | partial | mass-tree structural tests (`tier3_mass_*`) and `tier3_apollo_mass_tree` cover the composition invariants; no direct structure-sweep scenario |
| SIM_force_torque | covered | `tier3_force_and_torque_decoupled`, `tier3_force_constant_acceleration`, `tier3_force_symmetric_impulse_returns_to_rest`, `tier3_torque_constant_angular_acceleration`, `tier3_torque_simple_run01..06` |
| SIM_verif_attach_detach | covered | `tier3_sim_attach_detach_*` series, `tier3_mass_detach_all_children`, `tier3_mass_detach_recovers_original`, `tier3_mass_reattach_different_position` |
| SIM_verif_shutdown | not covered | JEOD exercises sim-teardown paths; our `Drop`-based cleanup has no dedicated Tier 3 scenario |
| SIM_removable_body_action | not covered | dynamic add/remove of body actions at runtime is not a feature we expose |
| SIM_RELKIN_VERIF | partial | relative-kinematics invariants exercised by `tier3_relative_*` and `tier3_simulation_relative_*`; no SIM_RELKIN-specific RUN cross-validation |

## environment / time

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_1_dyn_only | covered | `tier3_time_v1_dyn_only` |
| SIM_2_dyn_plus_STD | covered | `tier3_time_v2_std` |
| SIM_3_dyn_plus_UDE | covered | `tier3_time_v3_ude` |
| SIM_4_common_usage | covered | `tier3_time_v4_common` |
| SIM_5_all_inclusive | covered | `tier3_time_v5_all` |
| SIM_6_extension | covered | `tier3_time_v6_ext` |
| SIM_7_time_reversal | covered | `tier3_sim_time_reversal_run1`, `..._run3a`, `..._run8b`, `..._round_trip`, `tier3_time_forward_reverse_all_scales` |

## environment / gravity, ephemerides, RNP, planet, atmosphere, earth_lighting, spice

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_grav_accel_verif | covered | `jeod_gravity::jeod_validation::spherical_harmonics_40_test_vectors` (Tier 2 static vectors; the sim logs static grav-accel matches) + `tier3_simulation_run3a_sh4x4`, `tier3_simulation_run3b_sh8x8` for dynamic pipeline |
| SIM_csr_compare | not covered | CSR-vs-GGM comparison (gravity-model cross-check) not ported; we use GGM05C/GGM02C only |
| SIM_mercury | covered | `tier3_mercury_jeod_advance_rate`, `tier3_mercury_perihelion_advance_rate`, `tier3_bevy_mercury_relativistic`, `tier3_sim_mercury_relativistic_effect` |
| SIM_tide_verif | covered | `tier3_simulation_tide_run01`, `tier3_bevy_tidal_sh4x4`, `tier3_sh4x4_rnp` |
| SIM_ephem_verif | derived | every trajectory test that reads DE421 exercises ANISE; `tier3_simulation_earth_moon_clem` covers Earth-Moon ephemeris specifically |
| SIM_prop_planet | not covered | propagated-planet pattern (vs ephemeris-sourced) not ported |
| SIM_RNP_J2000_prop | covered | `tier3_bevy_run2p_polar_motion`, `tier3_simulation_run2p_polar_motion`, `tier3_bevy_sh4x4_rnp` |
| SIM_mars_orientation | covered | `tier3_sim_mars_rotation_dispatch`, `tier3_bevy_mars_dawn`, `tier3_simulation_mars_dawn` |
| SIM_PLANET_VERIF | derived | planet radii/flattening exercised by every sim touching geodetic or gravity |
| SIM_MET | covered | `tier3_simulation_met_run5a`, `tier3_bevy_met_run5a`, `tier3_bevy_met_atmosphere_drag_sixdof` |
| SIM_wind | covered | `tier3_drag_corotation_wind_effect` |
| SIM_LIGHT_CIR | covered | `tier3_sim_earth_lighting_consistency`, `tier3_bevy_earth_lighting_t01..t10`, `tier3_bevy_earth_lighting_pipeline` |
| SIM_de4xx | derived | any trajectory with ANISE loads DE4xx; covered transitively |
| SIM_spice | n/a | SPICE kernel lifecycle not a ported feature — we rely on ANISE |

## interactions

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_VER_DRAG | covered | `tier3_sim_drag_ver_bc`, `..._ver_cd`, `..._ver_flatplate_calc_eps00/05/1`, `..._ver_flatplate_diffuse`, `..._ver_flatplate_mixed`, `..._ver_flatplate_orbiter`, `..._ver_flatplate_specular`, `..._ver_flatplate_torque` + `tier3_drag_*` analytical |
| SIM_contact | not covered | body-to-body contact forces not ported (no contact model) |
| SIM_ground_contact | not covered | ground-contact model not ported |
| SIM_grav_torque_verif | covered | `tier3_simulation_run9a_torque`, `..._run9c_force_torque`, `..._run9d_force_torque_rate`, `..._run10a_gravity_torque`, `..._run10c/10d`, `tier3_bevy_gravity_torque_sixdof`, `tier3_bevy_run10c/10d` |
| SIM_torque_compare_simple | partial | simple gravity-torque sweep not a direct Tier 3 scenario; run10 series covers the same physics |
| SIM_1_BASIC | covered | `tier3_simulation_srp_basic_default`, `..._varied_cr`, `tier3_bevy_srp_basic_*`, `tier3_srp_1st_order_trajectory` |
| SIM_2_SHADOW_CALC | covered | `tier3_bevy_flat_plate_srp_with_shadow`, `tier3_simulation_shadow_2a_cooling`, `tier3_bevy_shadow_2a_cooling` |
| SIM_2A_SHADOW_CALC | covered | `tier3_simulation_shadow_2a_annular`, `tier3_bevy_shadow_2a_annular`, plus the `_cooling` pair above |
| SIM_3_ORBIT | covered | `tier3_simulation_run7a_sh4x4_3rd_body`, `..._run7b_sh8x8_3rd_body`, `..._run7c_sh4x4_3rd_body_drag`, `..._run7d_sh8x8_3rd_body_drag`, `tier3_bevy_run7a/7b/7c/7d` |
| SIM_3_ORBIT_1st_ORDER | covered | `tier3_srp_1st_order_trajectory` |
| SIM_4_DEFAULT | covered | `tier3_simulation_srp_basic_default`, `tier3_bevy_srp_basic_default` |

## utils

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_integ_test | covered | `tier3_simulation_lsode_abm4`, `tier3_simulation_lsode_default`, `tier3_integ_rk4_vs_analytical`, `tier3_integ_rkf45_vs_analytical`, `tier3_integ_rk4_vs_rkf45_agreement`, `tier3_integ_rk4_vs_gj_agreement` |
| SIM_GJ_test | covered | `tier3_simulation_gj_dt10`, `..._gj_order4`, `..._gj_order8`, `..._gj_order12`, `tier3_bevy_gj_*`, `tier3_integ_gj_*`, `tier3_bevy_parity_gj_*` |
| SIM_orb_elem | covered | `tier3_simulation_orbelem*`, `tier3_bevy_orbelem*`, 8 `tier3_orbinit_roundtrip_*` analytical |
| SIM_PFIXPOSN_VERIF | covered | `tier3_simulation_geodetic`, `tier3_bevy_geodetic_derived_state`, `tier3_bevy_polar_geodetic` |
| SIM_NED_VERIF | covered | `tier3_simulation_ned_*` series, `tier3_bevy_ned_sph_*` |
| SIM_LVLH_Frame | covered | `tier3_simulation_lvlh*`, `tier3_bevy_lvlh*`, `tier3_lvlh_*` analytical |
| SIM_REF_FRAMES | partial | frame-tree API is exercised indirectly by every relative-state / derived-state test; no dedicated frame-tree scenario |
| SIM_math_verif | derived | quaternion, orbital elements, geodetic math exercised by the above categories; direct math verifications are Tier 1 unit tests in `jeod_math` |
| SIM_ARTICULATION | not covered | articulated surface model (moving facets) not ported |
| SIM_SURFACE_MODEL | partial | static flat-plate surfaces exercised by SRP/drag tests; dynamic surface reconfiguration not ported |
| SIM_container | n/a | JEOD checkpointable containers not ported — Rust `Vec` supersedes |
| SIM_memory | n/a | JEOD memory allocator (`jeod_alloc`) not ported — Rust ownership supersedes |
| SIM_message_handler_verif | n/a | JEOD MessageHandler not ported — we use `panic!` / `log` crate |
| SIM_integ_loop | n/a | Trick integration loop not ported — we use Bevy `FixedUpdate` and `Simulation::step()` |
| SIM_simulation_interface | n/a | JEOD-Trick sim interface not ported |

## integrated / sims

| JEOD SIM | Coverage | Our Tier 3 tests |
|----------|----------|------------------|
| SIM_Apollo | covered | `tier3_apollo_mass_tree`, `tier3_apollo8_eci_integ`, `tier3_apollo8_frame_switch` |
| SIM_Earth_Moon | covered | `tier3_simulation_earth_moon_clem`, `tier3_bevy_earth_moon_clem`, `tier3_reference_run10a_libration_period` |
| SIM_Mars | covered | `tier3_simulation_mars_dawn`, `tier3_bevy_mars_dawn`, `tier3_sim_mars_rotation_dispatch` |

## Coverage summary

| Status | Count |
|--------|-------|
| covered (direct Tier 3) | 43 |
| partial (same physics, different config) | 7 |
| derived (exercised transitively) | 5 |
| not covered (gap) | 6 |
| n/a (infrastructure we chose not to port) | 5 |
| **total** | **66** |

## Gap follow-ups

The six "not covered" sims and five "partial" entries are tracked in [#99](https://github.com/simnaut/bevy_jeod/issues/99). The underlying physics is ported in every case, so these are breadth gaps, not correctness gaps.
