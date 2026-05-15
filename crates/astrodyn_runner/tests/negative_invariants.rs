//! Negative tests for `Simulation`-level JEOD invariants.
//!
//! Each `#[should_panic]` test drives a specific misconfiguration through
//! the public `Simulation` API and asserts the diagnostic substring that
//! pins the invariant. Pairs with the `// JEOD_INV` tags at the
//! corresponding enforcement sites; see `docs/JEOD_invariants.md` for
//! the catalog.

use astrodyn::{
    AngularVelocity, BodyAttitude, BodyFrame, ContactFacet, ContactMaterial, GravityControl,
    GravityControls, GravityGradient, GravityModel, GravitySource, GravitySourceEntry,
    InertiaTensor, IntegratorType, JeodQuat, MassProperties as SimMassProperties,
    MassPropertiesTyped, Position, RootInertial, RotationalState, RotationalStateTyped, SelfRef,
    SimulationTime, StructuralFrame, TranslationalState, TranslationalStateTyped, VehicleConfig,
    Velocity,
};
use astrodyn_runner::Simulation;
use glam::DVec3;
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

fn trans_typed(t: &TranslationalState) -> TranslationalStateTyped<RootInertial> {
    TranslationalStateTyped::<RootInertial> {
        // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
        position: Position::<RootInertial>::from_raw_si(t.position),
        // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
        velocity: Velocity::<RootInertial>::from_raw_si(t.velocity),
    }
}

fn rot_typed(r: &RotationalState) -> RotationalStateTyped<SelfRef> {
    RotationalStateTyped::<SelfRef>::new(
        BodyAttitude::from_jeod_quat(r.quaternion),
        // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
        AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(r.ang_vel_body),
    )
}

fn mass_typed(mp: &SimMassProperties) -> MassPropertiesTyped<SelfRef> {
    MassPropertiesTyped::<SelfRef>::with_inertia(
        Mass::new::<kilogram>(mp.mass),
        // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
        InertiaTensor::<BodyFrame<SelfRef>>::from_dmat3_unchecked(mp.inertia),
        // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
        Position::<StructuralFrame<SelfRef>>::from_raw_si(mp.position),
    )
    .with_t_parent_this(mp.t_parent_this)
}

/// Build a tiny two-body `Simulation` suitable for negative tests of
/// pair-registration APIs. Bodies share an empty point-mass source so
/// that registration semantics — not physics — drive the test.
fn build_two_body_sim() -> (Simulation, usize) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    let inertial = sim.add_source(
        "InertialAnchor",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: Position::<RootInertial>::zero(),
            velocity: Velocity::<RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
    let mp = SimMassProperties::new(1.0);
    let body_trans = TranslationalState {
        position: DVec3::new(1.0, 0.0, 0.0),
        velocity: DVec3::ZERO,
    };
    let body_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let make_body = || VehicleConfig {
        trans: trans_typed(&body_trans),
        rot: Some(rot_typed(&body_rot)),
        mass: Some(mass_typed(&mp)),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                inertial,
                GravityGradient::Skip,
            )],
        },
        ..Default::default()
    };
    let a = sim.add_body(make_body());
    let _b = sim.add_body(make_body());
    (sim, a)
}

// JEOD_INV: IN.30 — contact pair bodies must be distinct (JEOD
// `unique_pair`). Registering the same index for both legs of the pair
// would otherwise lead to a body applying force to itself, with the
// equal-and-opposite reaction subtracting from the same `TotalForce` —
// a silent zero-force pair plus accumulated rounding-error noise.
#[test]
#[should_panic(expected = "body A and body B must be distinct")]
fn in_30_panics_on_self_pair() {
    let (mut sim, a) = build_two_body_sim();
    let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.0);
    let facet_a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
    let facet_b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
    sim.register_contact_pair(a, facet_a, a, facet_b);
}

// JEOD_INV: PF.06 — RNP refresh cadence must be finite and >= 0; a
// negative cadence would make the cache-reuse comparison `current -
// cached >= cadence` flip sign and cause every step to re-use a stale
// matrix forever. The setter rejects the misconfiguration up front.
#[test]
#[should_panic(expected = "cadence must be finite and >= 0")]
fn pf_06_panics_on_negative_cadence() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    sim.set_earth_rnp_refresh_cadence(-1.0);
}

// JEOD_INV: PF.06 — sibling test for the non-finite branch. NaN would
// silently disable the cadence comparison (every comparison with NaN is
// false), making the cache-reuse path unreachable; the assert names
// the misconfiguration at the entry point.
#[test]
#[should_panic(expected = "cadence must be finite and >= 0")]
fn pf_06_panics_on_nan_cadence() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    sim.set_earth_rnp_refresh_cadence(f64::NAN);
}
