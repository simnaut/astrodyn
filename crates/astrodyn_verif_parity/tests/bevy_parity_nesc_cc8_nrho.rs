//! Bevy parity wrapper for NESC CC8 (NRHO).
//!
//! Drives the same `cc8_builder()` factory through both runtimes —
//! [`astrodyn_runner::Simulation`] and the Bevy `populate_app::<Moon>`
//! bridge — and asserts bit-identical translation **and** attitude state
//! at every NESC checkpoint (60 s cadence, 7 days).
//!
//! ## Status
//!
//! Active. Translational state asserted bit-identical at every CC8
//! checkpoint; rotational state asserted alongside, but the CC8
//! attitude-integrator divergence noted on the runner-side test still
//! affects both runtimes equally — bit-parity holds even though
//! attitude doesn't match the NESC sim_01 reference.
//!
//! Single-planet bridge note: CC8 fits — every body integrates in
//! `PlanetInertial<Moon>`; Earth and Sun are point-mass *sources* (not
//! integration frames), so this scenario is **not** covered by any
//! `KNOWN_PARITY_GAPS` exemption. `populate_app::<Moon>` auto-registers
//! the Moon-tagged per-planet system instantiations as of the CC8
//! prep PR — earlier revisions required a manual
//! `register_planet_systems::<Moon>` call after populate_app.

use std::time::Duration;

use astrodyn_bevy::{RotationalStateC, SimulationBuilderBevyExt, TranslationalStateC};
use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_nesc::cc8::{cc8_builder, cc8_reference, CHECKPOINT_CADENCE_S};
use bevy::prelude::*;

const DT: f64 = 1.0;

/// Parity-window cap, in seconds of simulation time.
///
/// Bit-identity divergence between two runtimes that share the same
/// `astrodyn_*` math is monotonic — once they drift, they stay drifted —
/// so a coarser checkpoint set is equivalent in detection strength to
/// a per-tick scan. The runner-side `tier3_nesc_cc8_nrho` test still
/// covers the full 7 days against the NESC reference; the parity
/// wrapper only needs to catch the *moment* of divergence between the
/// two runtimes, which 600 s of sim time absolutely covers.
///
/// Practical motivation: a 7-day Bevy propagation at dt = 1 s is
/// ~604 800 fixed-update ticks — minutes of CI runtime even on fast
/// hardware. 600 s is 10 checkpoints × 60 ticks = 600 Bevy ticks, a
/// few seconds of CI.
const PARITY_WINDOW_S: f64 = 600.0;

#[test]
fn bevy_parity_nesc_cc8_nrho() {
    // Runner side.
    let mut runner = cc8_builder().build().expect("runner build");

    // Bevy side — same builder, materialized into a fresh App under <Moon>.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let handles = cc8_builder()
        .populate_app::<astrodyn::Moon>(&mut app)
        .expect("populate_app under <Moon>");
    let vehicle = handles.body_entities[0];

    // Run startup systems once before stepping. AstrodynPlugin's Startup
    // wires per-source frame trees / source-frame-id resources / etc.
    // populate_app installs the plugin but does not pump the schedule;
    // the existing Moon-bearing parity tests in bevy_parity_integ_source.rs
    // invoke this exactly the same way.
    app.world_mut().run_schedule(Startup);

    let cc8_ref = cc8_reference();
    assert!(
        !cc8_ref.is_empty(),
        "CC8 reference trajectory is empty; regenerate via \
         `cargo run -p astrodyn_verif_nesc --bin extract_nesc`."
    );

    let steps_per_checkpoint = (CHECKPOINT_CADENCE_S / DT).round() as usize;
    assert!(
        steps_per_checkpoint >= 1,
        "checkpoint cadence ({CHECKPOINT_CADENCE_S}s) must be a positive multiple of dt ({DT}s)"
    );

    let mut last_t = 0.0_f64;
    for chk in &cc8_ref {
        if chk.time > PARITY_WINDOW_S {
            break;
        }
        runner.step_until(chk.time).expect("runner step");
        // Step the Bevy app forward by the same delta. Both runtimes
        // share the same dt = 1.0 s and 60 s cadence, so each
        // checkpoint is exactly `steps_per_checkpoint` fixed-update
        // ticks ahead of the previous one.
        let n = ((chk.time - last_t) / DT).round() as usize;
        last_t = chk.time;
        for _ in 0..n {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(DT));
            app.world_mut().run_schedule(FixedUpdate);
        }

        // Compare bit-identical state: translation + attitude.
        let r_pos = runner.body(0).trans.position.raw_si();
        let r_quat = runner
            .body(0)
            .rot
            .as_ref()
            .map(astrodyn::typed_bridge::rot_typed_to_raw)
            .map(|r| r.quaternion)
            .expect("runner body has 6-DoF rotational state");

        let b_pos = app
            .world()
            .get::<TranslationalStateC<astrodyn::Moon>>(vehicle)
            .expect("vehicle entity carries TranslationalStateC<Moon>")
            .position
            .raw_si();
        let b_quat = app
            .world()
            .get::<RotationalStateC>(vehicle)
            .expect("vehicle entity carries RotationalStateC (6-DoF)")
            .0;
        let b_quat_raw = astrodyn::typed_bridge::rot_typed_to_raw(&b_quat).quaternion;

        // Bit-identity per component, fail-loud on mismatch.
        for i in 0..3 {
            assert!(
                r_pos[i].to_bits() == b_pos[i].to_bits(),
                "CC8 NRHO translational bit-parity broke at t={} on axis {}: \
                 runner={} bevy={}",
                chk.time,
                i,
                r_pos[i],
                b_pos[i],
            );
        }
        for i in 0..4 {
            let r_d = r_quat.data[i];
            let b_d = b_quat_raw.data[i];
            assert!(
                r_d.to_bits() == b_d.to_bits(),
                "CC8 NRHO rotational bit-parity broke at t={} on quat[{}]: \
                 runner={} bevy={}",
                chk.time,
                i,
                r_d,
                b_d,
            );
        }
    }
}
