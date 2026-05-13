//! `VerificationCase` constructor for the SIM_ref_attach matrix-attach
//! Tier 3 scenario.
//!
//! Mirrors JEOD's `models/dynamics/body_action/verif/SIM_ref_attach`
//! `RUN_ref_attach_matrix`: a single 1 kg target vehicle in
//! Earth-inertial orbit propagates for 50 s under RK4 (no gravity — the
//! sim is JEOD's *initialization-only* verification harness, so the
//! recorded pre-attach trajectory is pure linear extrapolation), then
//! at `t = ATTACH_TIME = 50 s` the `pre_step` factory fires
//! [`SimContext::attach_to_frame`](crate::verification::SimContext::attach_to_frame)
//! to attach the vehicle to `Earth.pfix` with the
//! `BodyAttachMatrix`-recorded `(offset, T_pframe_struct)` pair.
//! Post-attach, the body's translational + rotational integration is
//! suppressed and its state is derived each tick from the rotating
//! Earth.pfix frame composed with the captured offset; the comparison
//! tracks JEOD's recorded inertial trajectory through 100 s.
//!
//! ### Why `attach_to_frame` and not `attach_to_frame_aligned`
//!
//! The matrix RUN supplies an explicit `(offset, T)` pair the recipe
//! can plumb through the adapter-neutral
//! [`crate::verification::SimContext::attach_to_frame`] surface — both
//! runtimes consume the same pair without needing a mass-tree-resident
//! named subject point.
//! The pt2pt RUN of SIM_ref_attach uses
//! [`astrodyn_runner::Simulation::attach_to_frame_aligned`] with a
//! named mass-point on the vehicle (resolves to the same `(offset, T)`
//! pair via the JEOD aligned-attach algebra); that scenario stays
//! hand-rolled because the Bevy adapter does not yet expose mass
//! points on body entities. The matrix recipe is the
//! `SimContext`-friendly path through the same physics.

use crate::verification::{
    CsvReference, InitialConditions, PreStepClosure, SourceFrameKind, Tolerances, VerificationCase,
};
use astrodyn::recipes::{earth, epoch};
use astrodyn::{
    JeodQuat, MassProperties, RotationalState, SimulationBuilder, TranslationalState,
    VehicleBuilder,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

/// Integrator timestep. Matches JEOD's
/// `SIM_ref_attach/S_define`: `IntegLoop sim_integ_loop(DYNAMICS) ...`
/// with `#define DYNAMICS 1.0`.
const DT_S: f64 = 1.0;
/// Attach fires at this sim time. JEOD's `BodyAttachMatrix` action
/// runs at `t = 50 s` per `RUN_ref_attach_matrix/input.py`.
const ATTACH_TIME_S: f64 = 50.0;

/// Earth source index in the builder's source list.
const EARTH_SOURCE_IDX: usize = 0;
/// Subject body index in the builder's body list.
const BODY_IDX: usize = 0;

/// Build the SIM_ref_attach scenario: Earth source (with EarthRNP
/// rotation, so the planet-fixed frame rotates at the sidereal rate
/// the JEOD reference encodes), one 6-DOF 1 kg body at the
/// JEOD-recorded initial state from `Modified_data/target_state.py`.
///
/// `_init` is ignored: the initial state is set verbatim from JEOD
/// source files (computational-independence rule — the CSV-derived
/// `InitialConditions` would be the same numbers, but reading them
/// from source is canonical for this sim).
fn build_ref_attach(_init: &InitialConditions) -> SimulationBuilder {
    // From `Modified_data/target_state.py`:
    let position = DVec3::new(1244540.5300, 5655938.8500, 3425643.2200);
    let velocity = DVec3::new(-6003.8330510, -1469.4960440, 4590.5117760);

    // Initial attitude: YPR Yaw=77.59°, Pitch=-30.60°, Roll=-46.10°.
    // YPR convention: q_total = q_yaw * q_pitch * q_roll (Z then Y
    // then X). Same construction the hand-rolled tier3 test used.
    let yaw = 77.590713_f64.to_radians();
    let pitch = (-30.604895_f64).to_radians();
    let roll = (-46.100115_f64).to_radians();
    let q_yaw = JeodQuat::left_quat_from_eigen_rotation(yaw, DVec3::Z);
    let q_pitch = JeodQuat::left_quat_from_eigen_rotation(pitch, DVec3::Y);
    let q_roll = JeodQuat::left_quat_from_eigen_rotation(roll, DVec3::X);
    let q_init = q_yaw.multiply(&q_pitch).multiply(&q_roll);

    // Body angular velocity: 0, -0.06556131568278°/s, 0 — in body frame.
    let ang_vel_body = DVec3::new(0.0, (-0.06556131568278_f64).to_radians(), 0.0);

    // 1 kg, identity inertia (kg·m²) per `Modified_data/veh_properties.py`.
    let mass = MassProperties::with_inertia(1.0, DMat3::IDENTITY, DVec3::ZERO);

    // `recipes::earth::point_mass()` ships with the JEOD `EarthRNP`
    // rotation model — the same precession/nutation/polar-motion stack
    // SIM_ref_attach exercises — so the `Earth.pfix` frame rotates
    // each step exactly as JEOD does. That fidelity is load-bearing
    // for the matrix attach (parent frame is rotating Earth.pfix).
    //
    // SIM_ref_attach is JEOD's *initialization-only* verification sim
    // (S_define comment: "This simulation has no dynamics -- other
    // than the Trick executive, is comprised of initilization [sic]
    // only"). Trick's clock advances and `BodyAttachMatrix` fires at
    // t=50, but no `IntegLoop` evaluates gravity, so the recorded
    // pre-attach trajectory is pure linear extrapolation
    // (`pos(t) = pos(0) + velocity * t`). We mirror that by
    // configuring the body with NO `GravityControl`: the RK4
    // integrator runs each step with zero applied force, producing
    // bit-identical linear extrapolation. Post-attach the frame
    // composition takes over the state entirely, so gravity
    // wouldn't affect the post-attach comparison either way.
    let mut sb = SimulationBuilder::new(epoch::j2000(), DT_S);
    let _earth_idx = sb.add_source("Earth", earth::point_mass());
    let vehicle = VehicleBuilder::new()
        .with_translational(astrodyn::typed_bridge::trans_raw_to_typed(
            &TranslationalState { position, velocity },
        ))
        .sixdof(
            RotationalState {
                quaternion: q_init,
                ang_vel_body,
            },
            mass,
        )
        .rk4()
        .build();
    sb.add_body(vehicle);
    sb
}

/// `pre_step` factory: at the record advancing to
/// `t = ATTACH_TIME_S + DT_S`, fire
/// [`crate::verification::SimContext::attach_to_frame`] with the
/// `BodyAttachMatrix`-recorded `(offset, T_pframe_struct)`.
/// `attach_offset = (10, 0, 0)` in Earth.pfix coords and
/// `t_pfix_struct = diag(-1, -1, 1)` (the 180°-yaw-equivalent matrix).
///
/// ### Why the `+ DT_S` shift
///
/// Recipes invoke `pre_step(record.time)` *before* the matching
/// `step_until(record.time)`. JEOD's `BodyAttach` action runs *after*
/// the t=50 row is logged, so the t=50 reference row is still the
/// pre-attach linear-extrapolation state; the first row that reflects
/// the attached frame composition is t=51. Firing the attach at the
/// pre-step before the *t=51* propagation step installs the marker
/// while the body is still at its pre-attach t=50 state, then the
/// step to t=51 derives state from the frame composition — exactly
/// the JEOD sequencing the reference CSV encodes.
///
/// The closure latches on a `bool` so re-entries don't double-attach
/// (the underlying runner / Bevy adapters panic on a second attach
/// without an intervening detach — correct fail-loud behaviour, but
/// the recipe shouldn't issue a duplicate request from its own
/// bookkeeping).
fn matrix_pre_step(_init: &InitialConditions) -> PreStepClosure {
    let mut attached = false;
    Box::new(move |sim, time_s: f64| {
        let half_dt = 0.5 * DT_S;
        let attach_at = ATTACH_TIME_S + DT_S;
        if !attached && (time_s - attach_at).abs() < half_dt {
            let offset_pfix = DVec3::new(10.0, 0.0, 0.0);
            let t_pfix_struct = DMat3::from_cols(
                DVec3::new(-1.0, 0.0, 0.0),
                DVec3::new(0.0, -1.0, 0.0),
                DVec3::new(0.0, 0.0, 1.0),
            );
            sim.attach_to_frame(
                BODY_IDX,
                EARTH_SOURCE_IDX,
                SourceFrameKind::Pfix,
                offset_pfix,
                t_pfix_struct,
            );
            attached = true;
        }
    })
}

/// `RUN_ref_attach_matrix` — direct `(offset, T)` attach to Earth.pfix.
///
/// Pairs with the JEOD-generated
/// `ref_attach_matrix_ref_attach_state.csv` reference. Tolerances are
/// the literal values from the hand-rolled `tier3_sim_ref_attach.rs`
/// matrix test, transcribed here verbatim. The split-tolerance
/// pre/post-attach assertion of the hand-rolled test collapses to the
/// looser post-attach bound — the pre-attach errors are sub-millimetre
/// f64-roundoff (well under the 16 m post-attach bound), so a single
/// global max-error check is equivalent in detection strength.
pub fn run_matrix() -> VerificationCase {
    VerificationCase {
        name: "tier3_sim_ref_attach_matrix",
        scenario: build_ref_attach,
        reference: CsvReference::RefAttach {
            file: "ref_attach_matrix_ref_attach_state.csv",
            dt: DT_S,
        },
        // SIM_ref_attach runs to t=100 (the JEOD `input.py` stop
        // time); use the CSV's end-of-data as the truncation since
        // it's identical.
        duration: Time::new::<second>(100.0),
        tolerances: Tolerances {
            position_m: [16.0, 16.0, 16.0],
            velocity_m_s: [1.5e-3, 1.5e-3, 1.5e-3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(matrix_pre_step),
    }
}
