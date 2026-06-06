//! `VerificationCase` constructors for the SIM_GJ_test Gauss-Jackson
//! integrator verification family.
//!
//! Cross-validates our Gauss-Jackson (Störmer-Cowell) integrator against
//! JEOD's implementation on a circular orbit. JEOD's SIM_GJ_test uses an
//! artificial μ (5.76e14) and a tweaked initial state designed to exercise
//! the GJ corrector. Each variant differs in GJ order or `time_scale_factor`.
//!
//! Variants:
//! - [`gj_order8`] — baseline GJ order 8, dt=1 s
//! - [`gj_order4`] — GJ order 4, dt=1 s
//! - [`gj_order12`] — GJ order 12, dt=1 s
//! - [`gj_dt10`] — GJ order 8 with `time_scale_factor=10` (effective dt=10 s)
//!
//! The tier3 sibling `crates/astrodyn_verif_jeod/tests/tier3_sim_gj.rs`
//! is the runner-vs-JEOD oracle; the parity tests at
//! `crates/astrodyn_verif_parity/tests/bevy_parity_gj.rs` feed these
//! recipes into `VerificationCaseParityExt::run_and_assert_parity` for
//! the runner-vs-bevy half of the transitivity argument.
//!
//! ## Duration
//!
//! All four variants run the full JEOD reference (`SIM_GJ_test` logs
//! every 300 sim-seconds for 300 000 sim-seconds, i.e. ~83 hours; the
//! `dt10` variant logs every 30 sim-seconds for 30 000 sim-seconds).
//! Bit-identity is bandwidth-cheap (~ms per integration tick on the
//! Bevy side) and a longer baseline catches any time-conditioned logic
//! divergence — e.g. a corrector path that only fires after N steps —
//! that a 1000-second prefix would miss.

use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
use astrodyn::{
    default_leap_second_table, GaussJacksonConfig, GravityControl, GravityControls,
    GravityGradient, GravityModel, GravitySource, GravitySourceEntry, IntegratorType, Position,
    RootInertial, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use uom::si::f64::Time;
use uom::si::time::second;

/// Non-standard μ matching JEOD SIM_GJ_test (`input_common.py`). The
/// constant is artificial — the verification sim deliberately picks a
/// value distinct from any real planet to guarantee the comparison
/// exercises the integrator in isolation, not gravitational-model
/// matching.
pub const MU_GJ_TEST: f64 = 5.76e14;

/// JEOD reference initial state for SIM_GJ_test: r₀ = [9 000 000, 0, 0] m,
/// v₀ = [0, 8 000, 0] m/s. Matches `tier3_sim_gj.rs` and the bootstrap
/// flavors in `bevy_parity_gj.rs` that don't drive a `VerificationCase`
/// (so they share the IC literal rather than duplicating it).
pub fn gj_initial_state() -> TranslationalState {
    TranslationalState {
        position: glam::DVec3::new(9.0e6, 0.0, 0.0),
        velocity: glam::DVec3::new(0.0, 8000.0, 0.0),
    }
}

/// Build a SIM_GJ_test scenario with the given GJ config, sim-side `dt`,
/// and `time_scale_factor`. Translational dynamics only; central-body
/// point-mass gravity at `MU_GJ_TEST`.
fn build_gj_scenario(
    config: GaussJacksonConfig,
    sim_dt: f64,
    time_scale_factor: f64,
) -> SimulationBuilder {
    let mut time = SimulationTime::at_j2000(default_leap_second_table());
    time.set_scale_factor(time_scale_factor);
    let mut b = SimulationBuilder::new(time, sim_dt);
    let earth = GravitySourceEntry {
        source: GravitySource {
            mu: MU_GJ_TEST,
            model: GravityModel::PointMass,
        },
        position: Position::<RootInertial>::zero(),
        velocity: astrodyn::Velocity::<RootInertial>::zero(),
        t_inertial_pfix: None,
        rotation_model: astrodyn::RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
        marker_only: false,
    };
    let earth_idx = b.add_source("Earth", earth);
    b.add_body(VehicleConfig {
        // allowed: typed↔raw kernel-boundary lift (named-method
        // opt-in; see #397).
        trans: super::typed_helpers::trans_typed(&gj_initial_state()),
        integrator: IntegratorType::GaussJackson(config),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth_idx,
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("sim-gj-0")
    });
    b
}

fn build_gj_order4(_init: &InitialConditions) -> SimulationBuilder {
    build_gj_scenario(
        // JEOD-faithful warn-and-continue on synthetic short-trajectory probes
        // that occasionally trip the corrector convergence check (#485 C1).
        // The tier3 reference CSV is the source of truth; matching it requires
        // the warn-and-continue path that JEOD uses.
        GaussJacksonConfig::with_order(4).with_allow_non_convergence(true),
        1.0,
        1.0,
    )
}

fn build_gj_order8(_init: &InitialConditions) -> SimulationBuilder {
    build_gj_scenario(
        GaussJacksonConfig::with_order(8).with_allow_non_convergence(true),
        1.0,
        1.0,
    )
}

fn build_gj_order12(_init: &InitialConditions) -> SimulationBuilder {
    build_gj_scenario(
        GaussJacksonConfig::with_order(12).with_allow_non_convergence(true),
        1.0,
        1.0,
    )
}

fn build_gj_dt10(_init: &InitialConditions) -> SimulationBuilder {
    // sim_dt=1.0, time_scale_factor=10.0 → effective dyn-time dt=10 s
    // per integration step. CSV records every 30 sim-seconds in the
    // dt10 reference, so the parity test asserts at every 30 ticks.
    build_gj_scenario(
        GaussJacksonConfig::with_order(8).with_allow_non_convergence(true),
        1.0,
        10.0,
    )
}

/// SIM_GJ_test baseline: GJ order 8, dt=1 s, tsf=1.0.
pub fn gj_order8() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_gj_order8",
        scenario: build_gj_order8,
        // SIM_GJ_test uses a 7-column `time + pos + vel` log layout
        // shared with SIM_orbinit, so it dispatches through the
        // existing `OrbInit` loader.
        reference: CsvReference::OrbInit("integ_gj_gj.csv"),
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            position_m: [1.321e-4, 1.309e-4, 1e-10],
            velocity_m_s: [1.161e-7, 1.168e-7, 1e-13],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_GJ_test order 4 variant.
pub fn gj_order4() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_gj_order4",
        scenario: build_gj_order4,
        reference: CsvReference::OrbInit("integ_gj_order4_gj.csv"),
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            position_m: [3.860e-5, 3.900e-5, 1e-10],
            velocity_m_s: [3.447e-8, 3.434e-8, 1e-13],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_GJ_test order 12 variant.
pub fn gj_order12() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_gj_order12",
        scenario: build_gj_order12,
        reference: CsvReference::OrbInit("integ_gj_order12_gj.csv"),
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            position_m: [1.943e-4, 1.939e-4, 1e-10],
            velocity_m_s: [1.725e-7, 1.728e-7, 1e-13],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}

/// SIM_GJ_test dt10 variant: `sim_dt=1`, `time_scale_factor=10` →
/// effective dyn-time `dt=10` s.
pub fn gj_dt10() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_gj_dt10",
        scenario: build_gj_dt10,
        reference: CsvReference::OrbInit("integ_gj_dt10_gj.csv"),
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            position_m: [1.036e0, 1.034e0, 1e-10],
            velocity_m_s: [9.189e-4, 9.193e-4, 1e-13],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: None,
    }
}
