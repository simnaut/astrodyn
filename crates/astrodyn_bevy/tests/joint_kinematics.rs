//! Bevy integration test for [`astrodyn_bevy::systems::joint_kinematics_system`].
//!
//! Spawns a bare Bevy app with [`astrodyn_bevy::AstrodynPlugin`] and a single
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

use astrodyn_bevy::prelude::*;
use astrodyn_bevy::systems;
use bevy::prelude::*;
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

/// Build a bare Bevy app with `AstrodynPlugin` and a single frame entity
/// carrying the given joint spec. Returns the app and the spawned
/// entity. Bypasses the full vehicle / planet machinery — the joint
/// kinematics system only depends on `SimulationTimeR` (advanced by
/// `time_advance_system`) and the per-entity [`JointKinematicsC`] /
/// [`FrameRotC`] / [`FrameAngVelC`] components.
fn build_app_with_joint(spec: JointKinematicsSpec) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
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
    app.add_plugins(AstrodynPlugin);

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

/// The system runs in [`AstrodynSet::EphemerisUpdate`] alongside
/// `planet_fixed_rotation_system`. A direct call into the `astrodyn`
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
    let (q_kernel, av_kernel) = astrodyn::evaluate_joint_kinematics(&spec, elapsed);
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
/// re-add it without re-importing through the `AstrodynPlugin` umbrella.
#[test]
fn joint_kinematics_system_is_a_public_system_function() {
    fn _assert_is_system_fn() {
        // Compile-time check: the path `astrodyn_bevy::systems::joint_kinematics_system`
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

/// Mission code that pre-validates joint axes upstream of
/// [`evaluate_joint_kinematics`] needs to thread the kernel's exact
/// tolerance through `astrodyn_bevy`/`astrodyn` rather than hard-coding a
/// parallel literal that could drift. Per the three-layer rule, mission
/// crates depend only on `astrodyn_bevy` (and transitively `astrodyn`),
/// never on `astrodyn_dynamics` directly — so `AXIS_NORM_TOL` must be
/// reachable on both surfaces. This test pins both paths.
#[test]
fn axis_norm_tol_is_reachable_through_astrodyn_and_prelude() {
    // Reachable through `astrodyn` (the single API surface for any
    // `astrodyn` consumer — astrodyn_bevy, astrodyn_runner, mission crates).
    let from_astrodyn: f64 = astrodyn::AXIS_NORM_TOL;
    // Reachable through the astrodyn_bevy prelude (the path mission code
    // actually uses: `use astrodyn_bevy::prelude::*;`).
    let from_prelude: f64 = AXIS_NORM_TOL;
    // Both must reference the same constant the kernel asserts against.
    assert_eq!(from_astrodyn, from_prelude);
    assert_eq!(from_astrodyn, astrodyn::AXIS_NORM_TOL);
}

/// Frame-tree integration: a `RelativeFrameState` walk that crosses a
/// joint frame must compose the parent frame's non-identity attitude /
/// angular velocity with the joint's per-tick kinematic update and
/// agree with the analytical answer.
///
/// This is the load-bearing check that `JointKinematicsC` is a
/// well-formed frame-tree consumer: spawning a joint frame leaves no
/// frame-tree component undefined (the `#[require]` triplet covers
/// `FrameTransC` / `FrameRotC` / `FrameAngVelC`), and the per-tick
/// rewrite of `FrameRotC` / `FrameAngVelC` flows through hierarchy
/// walks the same way `planet_fixed_rotation_system`'s output does.
///
/// Topology: `root_frame` → `parent_frame` → `joint_frame`.
///   - `root_frame`: identity triplet, no `ChildOf`.
///   - `parent_frame`: non-identity rotation about +X by `θ_p`,
///     non-zero angular velocity in parent-frame coords, zero
///     translation.
///   - `joint_frame`: `JointKinematicsC` rotating about +Y at a
///     constant rate; auto-inserted triplet from `#[require]`.
#[test]
fn joint_kinematics_relative_frame_state_walk_matches_analytical() {
    use astrodyn::{RefFrameRot, RefFrameState, RefFrameTrans};

    // Parent rotates about +X by θ_p; constant non-zero angular
    // velocity in this-frame coordinates so a frame-tree consumer that
    // skipped the parent's contribution would produce a visibly wrong
    // composed angular velocity.
    let theta_p = 0.7_f64;
    let parent_q = JeodQuat::left_quat_from_eigen_rotation(theta_p, DVec3::X);
    let parent_t = parent_q.left_quat_to_transformation();
    let parent_ang_vel = DVec3::new(0.05, -0.02, 0.03);

    // Joint rotates about +Y at a constant rate.
    let joint_rate = 0.4_f64;
    let joint_axis = DVec3::Y;
    let joint_initial = 0.1_f64;
    let spec = JointKinematicsSpec {
        axis_in_parent: joint_axis,
        rate_rad_per_s: joint_rate,
        initial_angle_rad: joint_initial,
    };

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let root_e = app
        .world_mut()
        .spawn((
            Name::new("root"),
            FrameTransC::default(),
            FrameRotC::default(),
            FrameAngVelC::default(),
        ))
        .id();

    let parent_e = app
        .world_mut()
        .spawn((
            Name::new("parent"),
            FrameTransC::default(),
            FrameRotC {
                q_parent_this: parent_q,
                t_parent_this: parent_t,
            },
            FrameAngVelC(parent_ang_vel),
            ChildOf(root_e),
        ))
        .id();

    let joint_e = app
        .world_mut()
        .spawn((
            Name::new("joint"),
            JointKinematicsC(spec),
            ChildOf(parent_e),
        ))
        .id();

    // Advance several ticks so the joint angle is visibly non-zero
    // and any silent identity / zero in the joint's frame state would
    // diverge from the analytical answer.
    let n_ticks = 5_usize;
    for _ in 0..n_ticks {
        step_once(&mut app);
    }

    // ── Analytical: compose the parent's RefFrameState (root → parent)
    //    with the joint's RefFrameState (parent → joint) using the
    //    same `incr_right` math `RelativeFrameState` walks under the
    //    hood. The joint's per-tick state is `(q_parent_joint(t),
    //    rate · axis)` from `evaluate_joint_kinematics`. ──
    let elapsed = app.world().resource::<SimulationTimeR>().tai_seconds;
    let (q_parent_joint, ang_vel_joint_in_joint) =
        astrodyn::evaluate_joint_kinematics(&spec, elapsed);
    let t_parent_joint = q_parent_joint.left_quat_to_transformation();

    let parent_state = RefFrameState {
        trans: RefFrameTrans {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: RefFrameRot {
            q_parent_this: parent_q,
            t_parent_this: parent_t,
            ang_vel_this: parent_ang_vel,
        },
    };
    let joint_state_rel_parent = RefFrameState {
        trans: RefFrameTrans {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: RefFrameRot {
            q_parent_this: q_parent_joint,
            t_parent_this: t_parent_joint,
            ang_vel_this: ang_vel_joint_in_joint,
        },
    };
    let expected_root_to_joint = parent_state.incr_right(&joint_state_rel_parent);

    // ── ECS read: `RelativeFrameState::relative_state(root, joint)`
    //    walks the ChildOf hierarchy and composes the per-node
    //    `FrameTransC` / `FrameRotC` / `FrameAngVelC` triplets. If the
    //    joint is missing any of the three, the walk panics with a
    //    diagnostic — proving the `#[require]` triplet contract. ──
    let actual_root_to_joint = app
        .world_mut()
        .run_system_cached_with(
            |In((from, to)): In<(Entity, Entity)>,
             rel: astrodyn_bevy::frame_param::RelativeFrameState|
             -> RefFrameState { rel.relative_state(from, to) },
            (root_e, joint_e),
        )
        .expect("run_system_cached_with should succeed");

    // Tolerance is the same per-component bound the other tests use —
    // both sides do exactly the same float arithmetic so any
    // disagreement signals a structural bug, not numeric drift.
    for i in 0..4 {
        let a = actual_root_to_joint.rot.q_parent_this.data[i];
        let e = expected_root_to_joint.rot.q_parent_this.data[i];
        assert!(
            (a - e).abs() < TOL,
            "composed q_parent_this[{i}]: ECS = {a}, analytical = {e}, diff {}",
            a - e
        );
    }
    for i in 0..3 {
        let a = actual_root_to_joint.rot.ang_vel_this[i];
        let e = expected_root_to_joint.rot.ang_vel_this[i];
        assert!(
            (a - e).abs() < TOL,
            "composed ang_vel_this[{i}]: ECS = {a}, analytical = {e}, diff {}",
            a - e
        );
    }
    for i in 0..3 {
        let a = actual_root_to_joint.trans.position[i];
        let e = expected_root_to_joint.trans.position[i];
        assert!(
            (a - e).abs() < TOL,
            "composed trans.position[{i}]: ECS = {a}, analytical = {e}, diff {}",
            a - e
        );
        let a = actual_root_to_joint.trans.velocity[i];
        let e = expected_root_to_joint.trans.velocity[i];
        assert!(
            (a - e).abs() < TOL,
            "composed trans.velocity[{i}]: ECS = {a}, analytical = {e}, diff {}",
            a - e
        );
    }

    // Direct cross-check: walking from `parent` to `joint` should
    // give exactly the joint's per-node state, since the parent is
    // the joint's immediate `ChildOf` ancestor.
    let parent_to_joint = app
        .world_mut()
        .run_system_cached_with(
            |In((from, to)): In<(Entity, Entity)>,
             rel: astrodyn_bevy::frame_param::RelativeFrameState|
             -> RefFrameState { rel.relative_state(from, to) },
            (parent_e, joint_e),
        )
        .expect("run_system_cached_with should succeed");
    for i in 0..4 {
        let a = parent_to_joint.rot.q_parent_this.data[i];
        let e = q_parent_joint.data[i];
        assert!(
            (a - e).abs() < TOL,
            "parent→joint q[{i}]: ECS = {a}, expected = {e}"
        );
    }
    for i in 0..3 {
        let a = parent_to_joint.rot.ang_vel_this[i];
        let e = ang_vel_joint_in_joint[i];
        assert!(
            (a - e).abs() < TOL,
            "parent→joint ang_vel[{i}]: ECS = {a}, expected = {e}"
        );
    }
}

// ===========================================================================
// Bevy integration tests for the enriched kinematic-only spec catalogue.
//
// These mirror the existing constant-rate tests above (single entity, single
// system, FixedUpdate ticked manually) for the three new spec shapes:
// sinusoidal, closure, multi-DOF. Each test bypasses the full vehicle /
// planet machinery — the joint-kinematics systems only depend on
// `SimulationTimeR` (advanced by `time_advance_system`) plus the per-entity
// spec component and the `FrameRotC` / `FrameAngVelC` storage triplet.
// ===========================================================================

/// Build a bare Bevy app carrying a single sinusoidal joint frame
/// entity. Mirrors `build_app_with_joint` for the sinusoidal spec.
fn build_app_with_sinusoidal_joint(spec: SinusoidalJointKinematicsSpec) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    let entity = app.world_mut().spawn(SinusoidalJointKinematicsC(spec)).id();
    (app, entity)
}

/// On each tick, the sinusoidal-driven joint frame's `FrameRotC` and
/// `FrameAngVelC` must agree (per-component, within float-trig
/// precision) with a direct kernel call at the same elapsed time.
/// Sampling N ticks across a non-trivial section of the period
/// catches sign / parameter-ordering bugs that would only surface
/// past `t = 0`.
#[test]
fn sinusoidal_joint_kinematics_matches_kernel_each_tick() {
    let spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        amplitude_rad: 0.3,
        omega_rad_per_s: 0.05,
        phase_rad: 0.4,
        offset_rad: 0.1,
    };
    let (mut app, entity) = build_app_with_sinusoidal_joint(spec);

    let n_ticks = 30_usize;
    for _ in 1..=n_ticks {
        step_once(&mut app);
        let elapsed = app.world().resource::<SimulationTimeR>().tai_seconds;
        let (q_kernel, av_kernel) = astrodyn::evaluate_sinusoidal_kinematics(&spec, elapsed);
        let (q, av) = read_rot_ang_vel(&app, entity);
        for i in 0..4 {
            let qa = q.data[i];
            let qe = q_kernel.data[i];
            assert!(
                (qa - qe).abs() < TOL,
                "elapsed={elapsed}: q[{i}] drift exceeds TOL: \
                 got {qa}, expected {qe}, diff {}",
                qa - qe
            );
        }
        for i in 0..3 {
            assert!(
                (av[i] - av_kernel[i]).abs() < TOL,
                "elapsed={elapsed}: ang_vel[{i}] drift exceeds TOL: \
                 got {}, expected {}",
                av[i],
                av_kernel[i]
            );
        }
    }
}

/// At the peak of the sinusoid (`t · ω + phase = π/2`) the angle
/// equals `offset + amplitude` and the rate is zero — closed-form
/// snapshot that's distinct from `t = 0`. Tick the schedule until
/// the peak is straddled and verify the angular velocity passes
/// through zero at that point.
#[test]
fn sinusoidal_joint_angular_velocity_zero_at_peak() {
    // Pick parameters so the peak lands exactly on a tick boundary
    // (no fractional-tick interpolation). With phase = 0 the peak is
    // at `ω · t = π/2` ⇒ `t = π/(2ω)`. With ω = π / (2 · DT) the
    // first peak is at t = DT, i.e. tick 1.
    let omega = std::f64::consts::FRAC_PI_2 / DT;
    let amplitude = 0.5_f64;
    let spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: amplitude,
        omega_rad_per_s: omega,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let (mut app, entity) = build_app_with_sinusoidal_joint(spec);

    step_once(&mut app);
    let (q, av) = read_rot_ang_vel(&app, entity);
    // Angle at peak = amplitude.
    let expected_q = JeodQuat::left_quat_from_eigen_rotation(amplitude, DVec3::Y);
    for i in 0..4 {
        assert!(
            (q.data[i] - expected_q.data[i]).abs() < 1.0e-12,
            "peak q[{i}]: got {}, expected {}",
            q.data[i],
            expected_q.data[i]
        );
    }
    // Rate at peak = amplitude · ω · cos(π/2) ≈ 0.
    for i in 0..3 {
        assert!(
            av[i].abs() < 1.0e-12,
            "peak ang_vel[{i}]: got {}, expected 0",
            av[i]
        );
    }
}

/// The closure-driven joint frame is constant in time — sampling at
/// multiple ticks must produce the same `FrameRotC` (bit-identical
/// each tick — the kernel's `JeodQuat::left_quat_from_eigen_rotation`
/// is deterministic in `(angle, axis)`) and a zero `FrameAngVelC`.
#[test]
fn closure_joint_kinematics_is_time_invariant_across_ticks() {
    let spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::new(1.0, 1.0, 1.0).normalize(),
        fixed_angle_rad: 0.42,
    };
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    let entity = app.world_mut().spawn(ClosureJointKinematicsC(spec)).id();

    step_once(&mut app);
    let (q1, av1) = read_rot_ang_vel(&app, entity);
    for _ in 0..10 {
        step_once(&mut app);
        let (q, av) = read_rot_ang_vel(&app, entity);
        for i in 0..4 {
            assert_eq!(
                q.data[i].to_bits(),
                q1.data[i].to_bits(),
                "closure q[{i}] not bit-identical across ticks"
            );
        }
        assert_eq!(av, av1);
        assert_eq!(av, DVec3::ZERO);
    }
    let expected = JeodQuat::left_quat_from_eigen_rotation(0.42, spec.axis_in_parent);
    assert_eq!(q1, expected);
}

/// 2-DOF chain: stage 0 constant-rate about Z, stage 1 sinusoidal
/// about Y. The Bevy adapter must dispatch the multi-DOF spec to the
/// kernel and write the composed `(rotation, angular velocity)` into
/// the entity's frame triplet. Comparison against a direct kernel
/// call at the same elapsed time pins the dispatch correctness, and
/// composing two qualitatively different kinematic styles in one
/// chain exercises the `evaluate_multi_dof` branch on a non-trivial
/// case.
#[test]
fn multi_dof_joint_kinematics_two_stage_matches_kernel() {
    let stage0 = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.4,
        initial_angle_rad: 0.1,
    };
    let stage1 = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: 0.25,
        omega_rad_per_s: 0.07,
        phase_rad: 0.3,
        offset_rad: 0.05,
    };
    let chain = MultiDofJointKinematicsSpec::from_slice(&[
        SingleDofKinematics::ConstantRate(stage0),
        SingleDofKinematics::Sinusoidal(stage1),
    ]);

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    let entity = app.world_mut().spawn(MultiDofJointKinematicsC(chain)).id();

    let n_ticks = 8_usize;
    for _ in 0..n_ticks {
        step_once(&mut app);
    }
    let elapsed = app.world().resource::<SimulationTimeR>().tai_seconds;
    let (q_kernel, av_kernel) = astrodyn::evaluate_multi_dof_kinematics(&chain, elapsed);
    let (q, av) = read_rot_ang_vel(&app, entity);
    for i in 0..4 {
        assert_eq!(
            q.data[i].to_bits(),
            q_kernel.data[i].to_bits(),
            "multi-DOF q[{i}] not bit-identical to kernel"
        );
    }
    for i in 0..3 {
        assert_eq!(
            av[i].to_bits(),
            av_kernel[i].to_bits(),
            "multi-DOF ang_vel[{i}] not bit-identical to kernel"
        );
    }
}

/// Each kinematic-spec component is *semantically alternative* —
/// each disjoint entity sets the same `FrameRotC` / `FrameAngVelC`
/// storage from its own driver. Spawning four entities, one carrying
/// each variant, must produce four distinct snapshots that each
/// match their respective kernel call. This guards against any
/// accidental cross-driver write ordering: if e.g. the sinusoidal
/// system queried `JointKinematicsC`-tagged entities by mistake, a
/// constant-rate entity's frame state would be overwritten with
/// sinusoidal output and the assertions would surface it.
#[test]
fn sibling_kinematic_drivers_dispatch_disjoint_entity_sets() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.5,
        initial_angle_rad: 0.0,
    };
    let sin_spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: 0.4,
        omega_rad_per_s: 0.1,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let close_spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::X,
        fixed_angle_rad: 0.7,
    };
    let multi_spec = MultiDofJointKinematicsSpec::from_slice(&[
        SingleDofKinematics::ConstantRate(const_spec),
        SingleDofKinematics::Closure(close_spec),
    ]);

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let const_e = app.world_mut().spawn(JointKinematicsC(const_spec)).id();
    let sin_e = app
        .world_mut()
        .spawn(SinusoidalJointKinematicsC(sin_spec))
        .id();
    let close_e = app
        .world_mut()
        .spawn(ClosureJointKinematicsC(close_spec))
        .id();
    let multi_e = app
        .world_mut()
        .spawn(MultiDofJointKinematicsC(multi_spec))
        .id();

    let n_ticks = 5_usize;
    for _ in 0..n_ticks {
        step_once(&mut app);
    }
    let elapsed = app.world().resource::<SimulationTimeR>().tai_seconds;

    // Each entity must match its own driver's kernel output.
    let (qc, avc) = read_rot_ang_vel(&app, const_e);
    let (qc_k, avc_k) = astrodyn::evaluate_joint_kinematics(&const_spec, elapsed);
    assert_eq!(qc, qc_k);
    assert_eq!(avc, avc_k);

    let (qs, avs) = read_rot_ang_vel(&app, sin_e);
    let (qs_k, avs_k) = astrodyn::evaluate_sinusoidal_kinematics(&sin_spec, elapsed);
    assert_eq!(qs, qs_k);
    assert_eq!(avs, avs_k);

    let (qcl, avcl) = read_rot_ang_vel(&app, close_e);
    let (qcl_k, avcl_k) = astrodyn::evaluate_closure_kinematics(&close_spec, elapsed);
    assert_eq!(qcl, qcl_k);
    assert_eq!(avcl, avcl_k);

    let (qm, avm) = read_rot_ang_vel(&app, multi_e);
    let (qm_k, avm_k) = astrodyn::evaluate_multi_dof_kinematics(&multi_spec, elapsed);
    assert_eq!(qm, qm_k);
    assert_eq!(avm, avm_k);

    // The four snapshots must not collapse onto each other (would
    // signal a query mis-tagging or a system overwriting another's
    // state). We check at least that each pair differs in at least
    // one quaternion component — distinct kernels with these
    // parameters cannot coincidentally match.
    let snapshots = [(qc, "const"), (qs, "sin"), (qcl, "close"), (qm, "multi")];
    for i in 0..snapshots.len() {
        for j in (i + 1)..snapshots.len() {
            let (qa, na) = snapshots[i];
            let (qb, nb) = snapshots[j];
            let differs = (0..4).any(|k| (qa.data[k] - qb.data[k]).abs() > TOL);
            assert!(
                differs,
                "{na} and {nb} produced indistinguishable rotations — \
                 a sibling system likely tagged the wrong entity set"
            );
        }
    }
}

/// A non-unit axis on the sinusoidal spec must panic at the kernel
/// boundary the first tick the system runs — same fail-loud
/// guarantee as the constant-rate spec.
#[test]
#[should_panic(expected = "must be a unit vector")]
fn sinusoidal_joint_kinematics_panics_on_non_unit_axis() {
    let spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
        amplitude_rad: 0.1,
        omega_rad_per_s: 1.0,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let (mut app, _entity) = build_app_with_sinusoidal_joint(spec);
    step_once(&mut app);
}

/// A non-unit axis on the closure spec must panic the first tick.
#[test]
#[should_panic(expected = "must be a unit vector")]
fn closure_joint_kinematics_panics_on_non_unit_axis() {
    let spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
        fixed_angle_rad: 0.5,
    };
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    app.world_mut().spawn(ClosureJointKinematicsC(spec));
    step_once(&mut app);
}

/// A non-unit axis on any DOF inside the multi-DOF chain must panic
/// the first tick — the kernel walks each stage and asserts at the
/// per-stage level.
#[test]
#[should_panic(expected = "must be a unit vector")]
fn multi_dof_joint_kinematics_panics_on_non_unit_axis_in_stage() {
    let bad_stage = JointKinematicsSpec {
        axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
        rate_rad_per_s: 1.0,
        initial_angle_rad: 0.0,
    };
    let chain =
        MultiDofJointKinematicsSpec::from_slice(&[SingleDofKinematics::ConstantRate(bad_stage)]);
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    app.world_mut().spawn(MultiDofJointKinematicsC(chain));
    step_once(&mut app);
}

/// Sibling joint-kinematics systems must be reachable by name from
/// outside the crate — mission code that wants a custom schedule
/// must be able to re-add them without re-importing through the
/// `AstrodynPlugin` umbrella, matching the existing
/// `joint_kinematics_system` contract.
#[test]
fn sibling_joint_kinematics_systems_are_public() {
    fn _assert_is_system_fn() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(Time::<Fixed>::from_seconds(DT));
        app.insert_resource(SimulationTimeR::default());
        app.add_systems(
            FixedUpdate,
            (
                systems::sinusoidal_joint_kinematics_system,
                systems::closure_joint_kinematics_system,
                systems::multi_dof_joint_kinematics_system,
            ),
        );
    }
    _assert_is_system_fn();
}

/// Capture the panic raised by `f` and downcast it to a `String`
/// suitable for substring assertions. Mirrors the helper pattern
/// used in `tests/validation_added_trigger.rs`. Returns the panic
/// message verbatim or fails the test if `f` did not panic.
fn capture_panic_message<F: FnOnce()>(f: F) -> String {
    use std::panic::AssertUnwindSafe;
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    let panic = result.expect_err(
        "expected the operation to panic with the joint-kinematics exclusivity diagnostic, \
         but it returned normally",
    );
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            panic
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
        })
        .unwrap_or_else(|| "<non-string panic payload>".to_string())
}

/// Assert that `msg` mentions every named component, the literal
/// "mutually exclusive" header, and the actionable "Fix:" tail.
/// `#[should_panic(expected = "...")]` only checks for a single
/// substring, so the multi-substring contract has to be expressed
/// explicitly in test code.
#[track_caller]
fn assert_diagnostic_lists(msg: &str, components: &[&str], entity: Entity) {
    assert!(
        msg.contains("mutually exclusive"),
        "panic message missing 'mutually exclusive' header: {msg}"
    );
    assert!(
        msg.contains("Fix:"),
        "panic message missing actionable 'Fix:' tail: {msg}"
    );
    let entity_str = format!("{entity:?}");
    assert!(
        msg.contains(&entity_str),
        "panic message did not name the offending entity {entity_str}: {msg}"
    );
    for name in components {
        assert!(
            msg.contains(name),
            "panic message did not mention {name}: {msg}"
        );
    }
}

/// Pull the entity-id token out of the `Offending entity: <token> carries [...]`
/// span of the diagnostic. Splits on the literal anchors, so a hook regression
/// that emits `Offending entity:  carries [...]` (whitespace only between the
/// two anchors) yields an empty token and trips
/// [`assert_entity_debug_shape`].
#[track_caller]
fn extract_offending_entity_token(msg: &str) -> String {
    let after_label = msg
        .split_once("Offending entity:")
        .unwrap_or_else(|| {
            panic!("panic message missing 'Offending entity:' phrase: {msg}");
        })
        .1;
    let (token_span, _) = after_label.split_once(" carries [").unwrap_or_else(|| {
        panic!("panic message missing ' carries [' tail after entity id: {msg}");
    });
    token_span.trim().to_string()
}

/// Assert `token` matches Bevy's `Entity` Debug formatting:
/// `{index}v{generation}` with non-empty digit runs on either side of the
/// single literal `'v'`. Catches the regression where the hook formats with
/// the wrong specifier, drops the id, or reports a placeholder.
#[track_caller]
fn assert_entity_debug_shape(token: &str, msg: &str) {
    assert!(
        !token.is_empty(),
        "expected an entity id token after 'Offending entity:', got empty string in: {msg}"
    );
    let (index, generation) = token.split_once('v').unwrap_or_else(|| {
        panic!("entity id token {token:?} is not in `{{index}}v{{generation}}` form: {msg}");
    });
    assert!(
        !index.is_empty() && index.chars().all(|c| c.is_ascii_digit()),
        "entity id index portion {index:?} of token {token:?} is not all digits: {msg}"
    );
    assert!(
        !generation.is_empty() && generation.chars().all(|c| c.is_ascii_digit()),
        "entity id generation portion {generation:?} of token {token:?} is not all digits: {msg}"
    );
}

/// Build a fully-wired Bevy app with `AstrodynPlugin` so the
/// joint-kinematics `on_insert` hooks and the `PostStartup`
/// validator are both installed.
fn build_app_with_plugin() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    app
}

/// An entity carrying two kinematic-spec components must be rejected
/// at insertion time. The four driver systems use `Without<...>`
/// filters for parallel scheduling, which would otherwise turn this
/// misconfiguration into a silent stale-state read; the fail-loud
/// `on_insert` hook prevents that. The diagnostic must include both
/// component names and the offending entity id.
///
/// Spawning a *single*-spec entity first and then inserting the
/// second spec via `entity_mut().insert(...)` lets the test capture
/// the entity id at spawn time so the substring assertion can verify
/// the exact id appears in the panic message.
#[test]
fn stacked_joint_specs_panic_at_insertion_two_specs() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let close_spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::X,
        fixed_angle_rad: 0.5,
    };
    let mut app = build_app_with_plugin();
    let entity = app.world_mut().spawn(JointKinematicsC(const_spec)).id();

    let msg = capture_panic_message(|| {
        app.world_mut()
            .entity_mut(entity)
            .insert(ClosureJointKinematicsC(close_spec));
    });

    assert_diagnostic_lists(
        &msg,
        &["JointKinematicsC", "ClosureJointKinematicsC"],
        entity,
    );
}

/// Two stacked specs *and* the entity id must appear in the
/// diagnostic. Builds the entity in stages so the test can capture
/// its id before the hook fires (a freshly-spawned entity carrying
/// only one spec is valid; inserting the second spec via
/// `entity_mut().insert(...)` is what trips the hook).
///
/// `#[should_panic(expected = "X")]` only checks one substring; this
/// test captures the panic and asserts every required substring —
/// the diagnostic header, the actionable "Fix:" tail, both
/// component names, and the exact offending entity id.
#[test]
fn stacked_joint_specs_diagnostic_names_all_components() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let sin_spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: 0.2,
        omega_rad_per_s: 0.05,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let mut app = build_app_with_plugin();
    let entity = app.world_mut().spawn(JointKinematicsC(const_spec)).id();

    // Insert the second spec — this must panic; the diagnostic must
    // mention both `JointKinematicsC` and
    // `SinusoidalJointKinematicsC` and the entity id captured above.
    let msg = capture_panic_message(|| {
        app.world_mut()
            .entity_mut(entity)
            .insert(SinusoidalJointKinematicsC(sin_spec));
    });
    assert_diagnostic_lists(
        &msg,
        &["JointKinematicsC", "SinusoidalJointKinematicsC"],
        entity,
    );
}

/// Three stacked specs landed in one bundle must list every name in
/// the diagnostic. Uses a fresh app and parses the entity id reported
/// by the hook out of the panic message itself.
///
/// We can't pre-allocate the entity via a separate `spawn` because
/// any *single* spec there would not panic (only stacking > 1 specs
/// does), and any prior bundle with two specs would already panic.
/// Instead we extract the token between `Offending entity: ` and
/// ` carries [` and assert it parses as a Bevy `Entity` Debug string
/// (`{index}v{generation}`). This catches regressions where the hook
/// drops the entity id from the diagnostic, which the weaker
/// `msg.contains("Offending entity:")` substring check would miss.
#[test]
fn stacked_joint_specs_diagnostic_names_three_components_in_one_bundle() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let sin_spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: 0.2,
        omega_rad_per_s: 0.05,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let close_spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::X,
        fixed_angle_rad: 0.5,
    };
    let mut app = build_app_with_plugin();

    let msg = capture_panic_message(|| {
        app.world_mut().spawn((
            JointKinematicsC(const_spec),
            SinusoidalJointKinematicsC(sin_spec),
            ClosureJointKinematicsC(close_spec),
        ));
    });

    assert!(
        msg.contains("mutually exclusive"),
        "panic missing diagnostic header: {msg}"
    );
    assert!(
        msg.contains("Fix:"),
        "panic missing actionable Fix tail: {msg}"
    );
    for name in [
        "JointKinematicsC",
        "SinusoidalJointKinematicsC",
        "ClosureJointKinematicsC",
    ] {
        assert!(
            msg.contains(name),
            "panic message did not mention {name}: {msg}"
        );
    }
    let id_token = extract_offending_entity_token(&msg);
    assert_entity_debug_shape(&id_token, &msg);
}

/// All four spec components on one entity must list all four names
/// in the diagnostic. Exercises the largest-fanout path through the
/// hook's name-collection branches. Same id-extraction strategy as
/// the three-spec test: parse the token between `Offending entity: `
/// and ` carries [` and verify it has Bevy's Entity Debug shape.
#[test]
fn stacked_joint_specs_diagnostic_names_all_four_components() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let sin_spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: 0.2,
        omega_rad_per_s: 0.05,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let close_spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::X,
        fixed_angle_rad: 0.5,
    };
    let multi_spec = MultiDofJointKinematicsSpec::from_slice(&[
        SingleDofKinematics::ConstantRate(const_spec),
        SingleDofKinematics::Closure(close_spec),
    ]);
    let mut app = build_app_with_plugin();

    let msg = capture_panic_message(|| {
        app.world_mut().spawn((
            JointKinematicsC(const_spec),
            SinusoidalJointKinematicsC(sin_spec),
            ClosureJointKinematicsC(close_spec),
            MultiDofJointKinematicsC(multi_spec),
        ));
    });

    for name in [
        "JointKinematicsC",
        "SinusoidalJointKinematicsC",
        "ClosureJointKinematicsC",
        "MultiDofJointKinematicsC",
    ] {
        assert!(
            msg.contains(name),
            "panic message did not mention {name}: {msg}"
        );
    }
    assert!(
        msg.contains("mutually exclusive"),
        "panic missing diagnostic header: {msg}"
    );
    assert!(msg.contains("Fix:"), "panic missing 'Fix:' tail: {msg}");
    let id_token = extract_offending_entity_token(&msg);
    assert_entity_debug_shape(&id_token, &msg);
}

/// A correctly-configured app — every kinematic spec on a distinct
/// entity — must pass both the `on_insert` hook and the
/// `PostStartup` validator. Guards against false positives that
/// would block legitimate multi-joint articulation chains.
#[test]
fn distinct_kinematic_entities_pass_startup_validation() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let sin_spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: 0.2,
        omega_rad_per_s: 0.05,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let close_spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::X,
        fixed_angle_rad: 0.5,
    };
    let multi_spec = MultiDofJointKinematicsSpec::from_slice(&[
        SingleDofKinematics::ConstantRate(const_spec),
        SingleDofKinematics::Closure(close_spec),
    ]);

    let mut app = build_app_with_plugin();
    app.world_mut().spawn(JointKinematicsC(const_spec));
    app.world_mut().spawn(SinusoidalJointKinematicsC(sin_spec));
    app.world_mut().spawn(ClosureJointKinematicsC(close_spec));
    app.world_mut().spawn(MultiDofJointKinematicsC(multi_spec));

    // Startup + PostStartup must not panic; the four entities each
    // carry a single spec.
    app.world_mut().run_schedule(Startup);
    app.world_mut().run_schedule(PostStartup);

    // A subsequent FixedUpdate tick should also succeed: the
    // `Without<...>` filters mean each driver writes its own
    // entity's storage and skips the others'.
    step_once(&mut app);
}

/// A user `Startup` system that spawns a stacked-spec entity via
/// `Commands` must trip the `on_insert` hook the moment the deferred
/// spawn is applied (during `Startup`'s command flush). Verifies
/// the diagnostic names every offending component, not just the one
/// the hook attributed the panic to.
#[test]
fn stacked_joint_specs_from_user_startup_commands_panic() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let close_spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::X,
        fixed_angle_rad: 0.5,
    };
    let mut app = build_app_with_plugin();
    app.add_systems(Startup, move |mut commands: Commands| {
        commands.spawn((
            JointKinematicsC(const_spec),
            ClosureJointKinematicsC(close_spec),
        ));
    });
    let msg = capture_panic_message(|| {
        // Running Startup is sufficient: deferred commands flush
        // during the schedule and the on_insert hook fires there.
        // Even if a future Bevy refactor moved the flush out of
        // `Startup`, `PostStartup` would still catch it.
        app.world_mut().run_schedule(Startup);
        app.world_mut().run_schedule(PostStartup);
    });
    assert!(
        msg.contains("JointKinematicsC") && msg.contains("ClosureJointKinematicsC"),
        "panic message did not name both stacked specs: {msg}"
    );
    assert!(
        msg.contains("mutually exclusive"),
        "panic message missing diagnostic header: {msg}"
    );
}

/// **Runtime-spawn regression test.** An entity spawned during
/// `FixedUpdate` (long after `Startup`/`PostStartup` have run) with
/// two kinematic specs must still trip the `on_insert` hook. This
/// covers the gap the original `PostStartup`-only validator left:
/// runtime spawns from `FixedUpdate` user systems would silently
/// drop out of every driver's `Without<...>` filter without the
/// hook.
///
/// The deferred `Commands::spawn` in the user system is applied at
/// the end of that system, which is when the hook fires — inside
/// `run_schedule(FixedUpdate)`. The test asserts the panic
/// propagates and the diagnostic names every offending component.
#[test]
fn stacked_joint_specs_from_runtime_fixed_update_panic() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let sin_spec = SinusoidalJointKinematicsSpec {
        axis_in_parent: DVec3::Y,
        amplitude_rad: 0.2,
        omega_rad_per_s: 0.05,
        phase_rad: 0.0,
        offset_rad: 0.0,
    };
    let mut app = build_app_with_plugin();

    // First flush Startup + PostStartup with no joint entities so
    // the validator passes. The runtime spawn happens later.
    app.world_mut().run_schedule(Startup);
    app.world_mut().run_schedule(PostStartup);

    // Single-shot FixedUpdate user system that spawns a stacked-spec
    // entity. Bevy applies the queued commands inside the schedule.
    app.add_systems(FixedUpdate, move |mut commands: Commands| {
        commands.spawn((
            JointKinematicsC(const_spec),
            SinusoidalJointKinematicsC(sin_spec),
        ));
    });

    let msg = capture_panic_message(|| {
        step_once(&mut app);
    });

    assert!(
        msg.contains("JointKinematicsC"),
        "runtime panic missing JointKinematicsC: {msg}"
    );
    assert!(
        msg.contains("SinusoidalJointKinematicsC"),
        "runtime panic missing SinusoidalJointKinematicsC: {msg}"
    );
    assert!(
        msg.contains("mutually exclusive"),
        "runtime panic missing diagnostic header: {msg}"
    );
}

/// **Runtime-insert regression test.** An entity that already
/// carries one kinematic spec must trip the `on_insert` hook the
/// moment a *second* spec is inserted on it via
/// `EntityCommands::insert` — even if that mutation happens in a
/// `Update` system long after the entity was first spawned. This
/// covers the late-mutation surface that a one-shot validator
/// cannot cover at all.
#[test]
fn late_insert_of_second_spec_panics_via_hook() {
    let const_spec = JointKinematicsSpec {
        axis_in_parent: DVec3::Z,
        rate_rad_per_s: 0.1,
        initial_angle_rad: 0.0,
    };
    let close_spec = ClosureJointKinematicsSpec {
        axis_in_parent: DVec3::X,
        fixed_angle_rad: 0.5,
    };
    let mut app = build_app_with_plugin();
    let entity = app.world_mut().spawn(JointKinematicsC(const_spec)).id();
    // Validator passes: only one spec on the entity.
    app.world_mut().run_schedule(Startup);
    app.world_mut().run_schedule(PostStartup);

    let msg = capture_panic_message(|| {
        // Direct world insertion, mirroring what `EntityCommands::insert`
        // would do once the deferred queue flushes.
        app.world_mut()
            .entity_mut(entity)
            .insert(ClosureJointKinematicsC(close_spec));
    });
    assert_diagnostic_lists(
        &msg,
        &["JointKinematicsC", "ClosureJointKinematicsC"],
        entity,
    );
}
