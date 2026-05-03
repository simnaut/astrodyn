//! Bevy integration test for [`bevy_jeod::systems::joint_kinematics_system`].
//!
//! Spawns a bare Bevy app with [`bevy_jeod::JeodPlugin`] and a single
//! frame entity carrying [`JointKinematicsC`], advances `FixedUpdate`
//! N ticks, and asserts after each tick that the entity's
//! [`FrameRotC`] / [`FrameAngVelC`] match the analytical answer:
//!
//! - rotation = left-transformation quaternion about
//!   `axis_in_parent` by `θ(t) = initial + rate · t`,
//! - angular velocity in this-frame coordinates = `rate · axis_in_parent`,
//! - both fields move strictly in lockstep with simulation time
//!   (driven by the JEOD-tracked `tai_seconds`, not Bevy's `Time<Fixed>`
//!   delta — the two should agree under the standard `time_advance_system`
//!   pipeline, but the kernel's contract is that it reads `SimulationTimeR`).

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::prelude::*;
use bevy_jeod::systems;
use glam::DVec3;

/// Simulation step (s).
const DT: f64 = 30.0;

/// Tolerance on per-component quaternion / angular-velocity comparisons.
/// The kernel reads `tai_seconds` which is built up by repeated `+= dt`
/// in `SimulationTime::advance`, so a few ULPs of drift versus the
/// closed-form `n * dt` reference are expected after many ticks.
const TOL: f64 = 1.0e-12;

fn step_once(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

/// Build a bare Bevy app with `JeodPlugin` and a single frame entity
/// carrying the given joint spec. Returns the app and the spawned
/// entity. Bypasses the full vehicle / planet machinery — the joint
/// kinematics system only depends on `SimulationTimeR` (advanced by
/// `time_advance_system`) and the per-entity [`JointKinematicsC`] /
/// [`FrameRotC`] / [`FrameAngVelC`] components.
fn build_app_with_joint(spec: JointKinematicsSpec) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    let entity = app
        .world_mut()
        // FrameTransC / FrameRotC / FrameAngVelC are auto-inserted via
        // the #[require(...)] attribute on JointKinematicsC, satisfying
        // the frame-tree triplet contract that RelativeFrameState walks
        // depend on.
        .spawn(JointKinematicsC(spec))
        .id();
    (app, entity)
}

fn read_rot_ang_vel(app: &App, entity: Entity) -> (JeodQuat, DVec3) {
    let rot = app.world().get::<FrameRotC>(entity).unwrap();
    let ang_vel = app.world().get::<FrameAngVelC>(entity).unwrap();
    (rot.q_parent_this, ang_vel.0)
}

/// On tick `n` (n ≥ 1) the joint angle is `initial + rate · n·dt` and
/// angular velocity is `rate · axis`.
#[test]
fn joint_kinematics_constant_rate_about_z_matches_analytical() {
    let rate = 10.0_f64.to_radians();
    let spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: rate,
        initial_angle_rad: 0.0,
    };
    let (mut app, entity) = build_app_with_joint(spec);

    // Pre-step: components should still be at the bundle defaults
    // (identity rot, zero ang_vel) because no FixedUpdate has run.
    let (q0, av0) = read_rot_ang_vel(&app, entity);
    assert_eq!(q0, JeodQuat::identity());
    assert_eq!(av0, DVec3::ZERO);

    let n_ticks = 60_usize; // 60 * 30 s = 30 minutes — well past π/2.
    for n in 1..=n_ticks {
        step_once(&mut app);
        let elapsed = (n as f64) * DT;
        let expected_angle = rate * elapsed;
        let expected_q = JeodQuat::left_quat_from_eigen_rotation(expected_angle, DVec3::Z);
        let (q, av) = read_rot_ang_vel(&app, entity);
        for i in 0..4 {
            let qa = q.data[i];
            let qe = expected_q.data[i];
            assert!(
                (qa - qe).abs() < TOL,
                "tick {n}: quaternion component {i} drift exceeds TOL: \
                 got {qa}, expected {qe}, diff {}",
                qa - qe
            );
        }
        let expected_av = DVec3::Z * rate;
        for i in 0..3 {
            assert!(
                (av[i] - expected_av[i]).abs() < TOL,
                "tick {n}: ang_vel component {i} drift exceeds TOL: \
                 got {}, expected {}",
                av[i],
                expected_av[i]
            );
        }
    }
}

/// Sign of `rate_rad_per_s` flips the angular-velocity vector.
#[test]
fn joint_kinematics_negative_rate_flips_angular_velocity() {
    let rate = -2.0_f64;
    let spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        rate_rad_per_s: rate,
        initial_angle_rad: 0.0,
    };
    let (mut app, entity) = build_app_with_joint(spec);

    step_once(&mut app);
    let (_q, av) = read_rot_ang_vel(&app, entity);
    let expected = DVec3::Y * rate;
    for i in 0..3 {
        assert!(
            (av[i] - expected[i]).abs() < TOL,
            "ang_vel component {i}: got {}, expected {}",
            av[i],
            expected[i]
        );
    }
}

/// `initial_angle_rad` shifts the angle so the first tick is
/// `initial + rate · dt`, not `rate · dt`.
#[test]
fn joint_kinematics_respects_initial_angle() {
    let theta0 = 0.5;
    let rate = 0.0; // Stationary at theta0.
    let spec = JointKinematicsSpec {
        axis_in_parent: DVec3::X,
        rate_rad_per_s: rate,
        initial_angle_rad: theta0,
    };
    let (mut app, entity) = build_app_with_joint(spec);

    step_once(&mut app);
    let (q, av) = read_rot_ang_vel(&app, entity);
    let expected_q = JeodQuat::left_quat_from_eigen_rotation(theta0, DVec3::X);
    for i in 0..4 {
        assert!(
            (q.data[i] - expected_q.data[i]).abs() < TOL,
            "quaternion component {i}: got {}, expected {}",
            q.data[i],
            expected_q.data[i]
        );
    }
    // Stationary joint ⇒ zero angular velocity regardless of initial angle.
    assert_eq!(av, DVec3::ZERO);
}

/// `t_parent_this` matrix on `FrameRotC` must agree with the quaternion
/// it was built from — the system writes both, and a downstream consumer
/// that reads only the matrix gets the same rotation as one that reads
/// the quaternion.
#[test]
fn joint_kinematics_writes_consistent_quat_and_matrix() {
    let spec = JointKinematicsSpec {
        axis_in_parent: DVec3::new(1.0, 1.0, 1.0).normalize(),
        rate_rad_per_s: 0.4,
        initial_angle_rad: 0.1,
    };
    let (mut app, entity) = build_app_with_joint(spec);

    for _ in 0..5 {
        step_once(&mut app);
    }
    let rot = app.world().get::<FrameRotC>(entity).unwrap();
    let mat_from_quat = rot.q_parent_this.left_quat_to_transformation();
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (rot.t_parent_this.col(c)[r] - mat_from_quat.col(c)[r]).abs() < 1.0e-15,
                "matrix row {r} col {c}: stored = {}, derived = {}",
                rot.t_parent_this.col(c)[r],
                mat_from_quat.col(c)[r]
            );
        }
    }
}

/// The system only touches entities that carry [`JointKinematicsC`].
/// A frame entity without the spec component must keep its default
/// (identity) FrameRotC / zero FrameAngVelC, even with the system
/// scheduled.
#[test]
fn joint_kinematics_does_not_touch_unrelated_frame_entities() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    // Spawn a bare frame entity (no JointKinematicsC).
    let entity = app
        .world_mut()
        .spawn((FrameRotC::default(), FrameAngVelC::default()))
        .id();

    // Also spawn one driven joint so the system has something to do.
    let driven = app
        .world_mut()
        .spawn(JointKinematicsC::new(DVec3::Z, 0.5, 0.0))
        .id();

    for _ in 0..3 {
        step_once(&mut app);
    }

    // Untagged entity stays at defaults.
    let (q, av) = read_rot_ang_vel(&app, entity);
    assert_eq!(q, JeodQuat::identity());
    assert_eq!(av, DVec3::ZERO);

    // Driven entity has rotated.
    let (qd, avd) = read_rot_ang_vel(&app, driven);
    assert_ne!(qd, JeodQuat::identity());
    assert_eq!(avd, DVec3::Z * 0.5);
}

/// The system runs in [`JeodSet::EphemerisUpdate`] alongside
/// `planet_fixed_rotation_system`. A direct call into the `jeod_sim`
/// kernel at the same simulation time must produce bit-identical
/// output — the Bevy adapter is a thin glue layer with no extra math.
#[test]
fn joint_kinematics_bevy_matches_kernel_bit_identical() {
    let spec = JointKinematicsSpec {
        axis_in_parent: DVec3::new(0.0, 1.0, 0.0),
        rate_rad_per_s: 1.7,
        initial_angle_rad: -0.3,
    };
    let (mut app, entity) = build_app_with_joint(spec);

    let n_ticks = 7_usize;
    for _ in 0..n_ticks {
        step_once(&mut app);
    }

    // Read SimulationTime as the kernel does.
    let elapsed = app.world().resource::<SimulationTimeR>().tai_seconds;
    let (q_kernel, av_kernel) = jeod_sim::evaluate_joint_kinematics(&spec, elapsed);
    let (q_bevy, av_bevy) = read_rot_ang_vel(&app, entity);
    for i in 0..4 {
        assert_eq!(
            q_bevy.data[i].to_bits(),
            q_kernel.data[i].to_bits(),
            "quaternion component {i} not bit-identical: \
             bevy = {} (bits = {:#018x}), kernel = {} (bits = {:#018x})",
            q_bevy.data[i],
            q_bevy.data[i].to_bits(),
            q_kernel.data[i],
            q_kernel.data[i].to_bits(),
        );
    }
    for i in 0..3 {
        assert_eq!(
            av_bevy[i].to_bits(),
            av_kernel[i].to_bits(),
            "ang_vel component {i} not bit-identical: \
             bevy = {} (bits = {:#018x}), kernel = {} (bits = {:#018x})",
            av_bevy[i],
            av_bevy[i].to_bits(),
            av_kernel[i],
            av_kernel[i].to_bits(),
        );
    }
}

/// A non-unit `axis_in_parent` must panic at the kernel boundary the
/// first tick the system runs. Fail-loud rule: silently rescaling the
/// rotation angle would produce a wrong-physics answer with no
/// diagnostic.
#[test]
#[should_panic(expected = "must be a unit vector")]
fn joint_kinematics_panics_on_non_unit_axis() {
    let spec = JointKinematicsSpec {
        axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
        rate_rad_per_s: 1.0,
        initial_angle_rad: 0.0,
    };
    let (mut app, _entity) = build_app_with_joint(spec);
    step_once(&mut app);
}

/// Minimal smoke test that `joint_kinematics_system` is reachable by
/// name from outside the crate — the function must remain a public
/// system function so mission code that wants a custom schedule can
/// re-add it without re-importing through the `JeodPlugin` umbrella.
#[test]
fn joint_kinematics_system_is_a_public_system_function() {
    fn _assert_is_system_fn() {
        // Compile-time check: the path `bevy_jeod::systems::joint_kinematics_system`
        // resolves and is callable as a Bevy system. Reachability proves the
        // symbol stays in the public surface.
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(Time::<Fixed>::from_seconds(DT));
        app.insert_resource(SimulationTimeR::default());
        app.add_systems(FixedUpdate, systems::joint_kinematics_system);
    }
    _assert_is_system_fn();
}
