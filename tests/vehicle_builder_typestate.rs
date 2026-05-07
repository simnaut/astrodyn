//! Runtime cover for the typestate `VehicleBuilder`.
//!
//! Compile-fail gating is locked in via `compile_fail` doctests on the
//! [`vehicle_builder`](astrodyn::vehicle_builder) module — those run via
//! `cargo test --doc -p astrodyn` and prove that out-of-order calls
//! produce a compile error rather than a runtime panic.
//!
//! This file exercises the **happy paths** at runtime:
//!
//! - `with_translational` → `three_dof_point_mass` → `rk4` → `build`
//! - `with_translational` → `sixdof` → `rkf45` → `build` with optional
//!   gravity / drag / SRP additions in the `Ready` state
//! - `from_orbital_elements` → … round-trips an ISS-class state through
//!   the typed entry path

use astrodyn::vehicle_builder::VehicleBuilder;
use astrodyn::vehicle_config::VehicleConfig;
use astrodyn::{
    Earth, GaussJacksonConfig, IntegratorType, PlanetInertial, Position, RootInertial, Velocity,
};
use astrodyn_dynamics::state::TranslationalStateTyped;
use astrodyn_dynamics::{MassProperties, RotationalState, TranslationalState};
use astrodyn_gravity::GravityControl;
use astrodyn_interactions::DragConfig;
use astrodyn_math::{JeodQuat, OrbitalElements};
use astrodyn_quantities::ext::F64Ext;
use glam::{DMat3, DVec3};

fn iss_trans() -> TranslationalStateTyped<RootInertial> {
    TranslationalStateTyped::<RootInertial>::from_untyped_unchecked(&TranslationalState {
        position: DVec3::new(6_778_000.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7_672.0, 0.0),
    })
}

#[test]
fn three_dof_rk4_round_trip() {
    let cfg: VehicleConfig = VehicleBuilder::new()
        .with_translational(iss_trans())
        .three_dof_point_mass(420_000.0.kg())
        .rk4()
        .build();
    assert_eq!(cfg.integrator, IntegratorType::Rk4);
    assert_eq!(cfg.mass.expect("mass set by typestate").mass, 420_000.0);
    assert!(cfg.rot.is_none());
    assert!(cfg.drag.is_none());
}

#[test]
fn six_dof_rkf45_with_options() {
    let rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let mass = MassProperties::with_inertia(420_000.0, DMat3::IDENTITY * 1.0e6, DVec3::ZERO);
    let drag = DragConfig {
        cd: 2.2,
        area: 40.0,
        constant_density: None,
    };
    let cfg = VehicleBuilder::new()
        .with_translational(iss_trans())
        .sixdof(rot, mass)
        .rkf45()
        .gravity(GravityControl::new_spherical(0, false))
        .drag(drag)
        .build();
    assert_eq!(cfg.integrator, IntegratorType::Rkf45);
    assert!(cfg.rot.is_some());
    assert_eq!(cfg.gravity_controls.controls.len(), 1);
    assert_eq!(cfg.drag.as_ref().unwrap().cd, 2.2);
}

#[test]
fn gauss_jackson_integrator_selection() {
    let cfg = VehicleBuilder::new()
        .with_translational(iss_trans())
        .three_dof_point_mass(1_000.0.kg())
        .gauss_jackson(GaussJacksonConfig::default())
        .build();
    assert!(matches!(cfg.integrator, IntegratorType::GaussJackson(_)));
}

#[test]
fn from_orbital_elements_round_trip() {
    let earth_mu = 3.986_004_415e14_f64.m3_per_s2();
    // The Bevy/builder API stores state in `RootInertial` (current sims have
    // root=Earth.inertial); orbital elements are computed in
    // `PlanetInertial<Earth>` — the planet of the gravitating body. Relabel
    // (bit-identical) at the call site.
    let pos = Position::<PlanetInertial<Earth>>::from_raw_si(iss_trans().position.raw_si());
    let vel = Velocity::<PlanetInertial<Earth>>::from_raw_si(iss_trans().velocity.raw_si());
    let oe = OrbitalElements::<Earth>::from_cartesian_typed(earth_mu, pos, vel)
        .expect("ISS-class state has well-defined orbital elements");

    let cfg = VehicleBuilder::new()
        .from_orbital_elements(oe.clone(), earth_mu)
        .three_dof_point_mass(420_000.0.kg())
        .rk4()
        .build();

    // The reconstructed translational state must round-trip the original
    // to within numerical tolerance — `from_orbital_elements` delegates
    // to `init_from_orbital_elements_typed`, itself delegating to the
    // bit-identical f64 implementation.
    let pos_err = (cfg.trans.position - iss_trans().position.raw_si()).length();
    let vel_err = (cfg.trans.velocity - iss_trans().velocity.raw_si()).length();
    assert!(pos_err < 1.0e-6, "position round-trip error: {pos_err}");
    assert!(vel_err < 1.0e-9, "velocity round-trip error: {vel_err}");
}
