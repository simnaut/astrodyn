//! Tier 3: SIM_dyncomp RUN_5A — MET atmosphere via Simulation pipeline
//!
//! Propagates an ISS-like elliptical orbit with point-mass gravity and MET
//! atmosphere through `Simulation::step()`, comparing atmosphere density
//! against JEOD reference trajectory at each checkpoint.
//!
//! RUN_5A: minimum solar activity (F10.7=70, Ap=0). Drag is disabled in JEOD's
//! RUN_5A config, so the atmosphere has no effect on the trajectory. We still
//! configure it to validate that our MET density computation matches JEOD's.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_atmosphere::met as met_atmosphere;
use jeod_runner::{GravitySourceEntry, RotationModel, SimBody, Simulation};
use jeod_sim::{
    AtmosphereConfig, AtmosphereModel, AtmosphereState, DynamicsConfig, GravityControl,
    GravityControls, GravityModel, GravitySource, JeodQuat, MetAtmosphere, RotationalState,
    SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

#[test]
fn tier3_simulation_met_run5a() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path("dyncomp_run5a_atmos_atmos_traj.csv");
    assert!(
        csv_path.exists(),
        "Atmosphere trajectory CSV not found at {}.\n\
         Generate with: docker run --rm -e FORCE=1 --entrypoint /bin/bash \
         -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
         jeod-trick /generate_references.sh",
        csv_path.display()
    );

    let sim_dir = jeod_root.join(SIM_DYNCOMP);
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");

    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let mu_earth =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth mu");

    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_iss"),
    );
    let mass_props = mass_props_from_init(&mass_init);

    let records = load_atmos_traj_csv(&csv_path);
    assert!(records.len() > 100, "Expected >100 records");
    let init = &records[0];

    // MET atmosphere: minimum solar activity (RUN_5A: F10.7=70, Ap=0)
    let met_model = MetAtmosphere {
        f10: 70.0,
        f10b: 70.0,
        geo_index: 0.0,
        geo_index_type: met_atmosphere::GeoIndexType::Ap,
    };

    // Load epoch from JEOD time config (matches SIM_dyncomp epoch)
    let time_cfg =
        jeod_test_data::time_config::load_time_config(&sim_dir.join("Modified_data/time.py"));
    let epoch_tai_tjt = time_cfg.tai_tjt();
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp requires UT1-TAI offset for GMST/RNP");
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(ut1_tai_offset);
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: mu_earth,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
    });

    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met_model),
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: OMEGA_EARTH,
    });
    sim.atmosphere_planet_source = Some(earth);

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, true)],
        },
        atmospheric_state: Some(AtmosphereState::default()),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_dyncomp RUN_5A MET atmosphere, {} points",
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut ref_states = Vec::with_capacity(records.len() - 1);
    let mut max_density_rel_err = 0.0_f64;

    for rec in &records[1..] {
        sim.step_until(rec.time);

        let body = sim.body(0);
        let atmos = body
            .atmospheric_state
            .as_ref()
            .expect("atmospheric_state should be computed");

        if rec.density.abs() > 1e-30 {
            let rel_err = ((atmos.density - rec.density) / rec.density).abs();
            max_density_rel_err = max_density_rel_err.max(rel_err);
        }

        our_states.push(StateLog {
            time: rec.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: rec.time,
            position: Some(rec.position),
            velocity: Some(rec.velocity),
            ..Default::default()
        });
    }

    let report = CrossvalReport::compute("tier3_simulation_met_run5a", &our_states, &ref_states);
    report.write();

    println!(
        "  Max position error: {:.6e} m",
        report.max_position_component()
    );
    println!("  Max density relative error: {:.6e}", max_density_rel_err);

    // Position: same gravity model, so trajectory should match closely
    report.assert_position([2.5e-6, 2.5e-6, 2.5e-6]);
    // MET density: small differences from position-dependent altitude at different
    // orbital phases and epoch-dependent time in the MET model tables.
    assert!(
        max_density_rel_err < 1.4e-3,
        "MET density relative error {max_density_rel_err:.3e} exceeds 1.4e-3"
    );
}
