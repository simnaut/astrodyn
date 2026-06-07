//! Registration-order freedom for identity-keyed gravity controls
//! (issue #668): `SimBody` stores uid-keyed config verbatim and every
//! consumption site resolves identities at step/validate time through
//! the simulation's uid → index boundary map — so a body added BEFORE
//! its third-body source still resolves once the source registers.
//! (The pre-#668 index keying had the same order freedom only because
//! indices were guessed-ahead integers; identity keying makes the
//! reference meaningful while preserving the freedom.)
//!
//! `integ_source` is the documented exception: it resolves at
//! `add_body` (the body's frame entity is parented at registration),
//! exactly as the index form did.

use astrodyn::{
    FrameUid, GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
    GravitySourceEntry, PlanetInertial, SimulationTime, VehicleConfig,
};
use astrodyn_runner::Simulation;
use glam::DVec3;

#[test]
fn body_added_before_source_resolves_at_step_time() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    let _earth = sim.add_source("Earth", GravitySourceEntry::central_body(&astrodyn::EARTH));

    // Body referencing the Moon by identity — the Moon is NOT
    // registered yet.
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&astrodyn::TranslationalState {
            position: DVec3::new(7.0e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7.5e3, 0.0),
        }),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(
                    FrameUid::of::<PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                ),
                GravityControl::new_third_body(FrameUid::of::<PlanetInertial<astrodyn::Moon>>()),
            ],
        },
        ..VehicleConfig::named("source-after-body")
    });

    // Now register the Moon. The body's control resolves through the
    // boundary map at step time, not at add_body.
    let _moon = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: astrodyn::MOON.shape.mu,
                model: GravityModel::PointMass,
            },
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(DVec3::new(3.844e8, 0.0, 0.0)),
            None,
        ),
    );

    assert!(
        sim.validate().is_ok(),
        "identity-keyed controls must validate regardless of body/source order"
    );
    sim.step_n(3).expect("step with late-registered source");
    let r = sim.body(0).trans.position.raw_si().length();
    assert!(r > 6.0e6, "body integrated under both sources: r = {r}");
}
