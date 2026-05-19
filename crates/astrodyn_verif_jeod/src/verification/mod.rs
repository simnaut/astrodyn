// The `recipes::verification` module is hidden from rendered rustdoc
// (declared in `recipes/mod.rs`) — the entire submodule is
// workspace-internal Tier 3 scaffolding that downstream mission code
// should not consume. Intra-doc links inside this file therefore
// aren't surfaced anywhere; allow the broken-link lint so we don't
// have to chase resolution that rustdoc suppresses for hidden
// modules.
#![allow(rustdoc::broken_intra_doc_links)]

//! Verification-case scaffolding.
//!
//! [`VerificationCase`] bundles a scenario constructor, a reference
//! CSV path, propagation duration, and per-component tolerances into a
//! single declarative unit. Tier 3 tests in Phase 7/8 collapse to:
//!
//! // reason: `run_and_assert` is defined by `crate::VerificationCaseExt`, which astrodyn cannot depend on without a circular workspace dependency.
//! ```ignore
//! #[test]
//! fn tier3_dyncomp_run2_3dof() {
//!     verification::dyncomp_run2_3dof().run_and_assert();
//! }
//! ```
//!
//! Phase 6 shipped only the scaffold: the [`VerificationCase`] /
//! [`Tolerances`] / [`CsvReference`] types and the
//! [`reference_data`] submodule for JEOD-source-dependent loaders
//! (gravity coefficient files, etc.). Phase 7 expands [`CsvReference`]
//! into a tagged enum that names the per-CSV layout and provides one
//! constructor per Tier 3 case in `sim_dyncomp` (and follow-on
//! family modules).
//!
//! `run_and_assert` itself is implemented by `astrodyn_runner` via an
//! extension trait, since materializing a [`SimulationBuilder`] into a
//! runtime `astrodyn_runner::Simulation` is runner-specific. The runner-side
//! trait also dispatches on the [`CsvReference`] variant, calling the
//! matching loader from `crate::tier3_csv`.

pub mod reference_data;

use glam::{DMat3, DQuat, DVec3};
use uom::si::f64::Time;

use astrodyn::SimulationBuilder;

/// Adapter-neutral interface for the operations a `pre_step` hook needs.
///
/// Implemented by `astrodyn_runner::Simulation` (and any future ECS adapter
/// that materializes a `VerificationCase`). Lets a `pre_step` closure
/// inject state into the running simulation between reference-CSV time
/// steps without depending on the `astrodyn_runner` crate.
pub trait SimContext {
    /// Set the inertial position of source `source_idx`.
    fn set_source_position(&mut self, source_idx: usize, position: DVec3);
    /// Set the inertial position and velocity of source `source_idx`.
    fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3);
    /// Update the inertial position of one tidal body inside source
    /// `source_idx`'s tidal configuration. Used by tide-validation
    /// hooks that drive Sun/Moon positions for the tidal ΔC20 each
    /// step. Panics if `source_idx` lacks a tidal config or
    /// `tidal_body_idx` is out of range — these are programmer errors,
    /// not runtime conditions, since the recipe wires the tidal
    /// config at construction time.
    ///
    /// The default implementation panics with an explicit
    /// "tidal bodies not supported" message so existing `SimContext`
    /// implementors stay source-compatible. Adapters that wire
    /// tidal-body state into a `Simulation`-equivalent should
    /// override this.
    fn set_tidal_body_position(
        &mut self,
        source_idx: usize,
        tidal_body_idx: usize,
        position: DVec3,
    ) {
        let _ = (source_idx, tidal_body_idx, position);
        panic!("tidal bodies not supported by this SimContext implementation");
    }

    /// Attach `child_idx` to `parent_idx` in the mass tree at `offset`
    /// (child's structural origin in the parent's structural frame, m)
    /// with `t_parent_child` (rotation from the parent's structural
    /// frame into the child's structural frame). Mirrors the runner's
    /// `Simulation::attach` runtime entry point — the implementation
    /// must run JEOD's momentum-conservation combine kernel and reset
    /// affected integrators so the post-attach state is bit-identical
    /// across runtimes. Used by mid-flight attach/detach scenarios
    /// that schedule topology changes via `pre_step`.
    ///
    /// The default implementation panics with an explicit
    /// "attach not supported" message so existing `SimContext`
    /// implementors stay source-compatible. Adapters that own a
    /// mass-tree mutation surface (the runner's `Simulation`, the
    /// Bevy adapter's `AttachEvent` bus) override this.
    fn attach(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        let _ = (child_idx, parent_idx, offset, t_parent_child);
        panic!(
            "runtime attach not supported by this SimContext implementation; \
             provide a SimContext impl that drives the adapter's mass-tree \
             attach path (e.g. AttachEvent on the Bevy bus)"
        );
    }

    /// Detach `child_idx` from its current parent in the mass tree.
    /// Mirrors the runner's `Simulation::detach` runtime entry point —
    /// the implementation must shift the parent's composite state by
    /// the inertial-frame CoM-delta and reset affected integrators so
    /// the post-detach state is bit-identical across runtimes.
    ///
    /// The default implementation panics with an explicit
    /// "detach not supported" message so existing `SimContext`
    /// implementors stay source-compatible. Adapters that own a
    /// mass-tree mutation surface override this.
    fn detach(&mut self, child_idx: usize) {
        let _ = child_idx;
        panic!(
            "runtime detach not supported by this SimContext implementation; \
             provide a SimContext impl that drives the adapter's mass-tree \
             detach path (e.g. DetachEvent on the Bevy bus)"
        );
    }

    /// Mark `child_idx` as a kinematic-only (non-integrated) child of
    /// its mass-tree parent. Mirrors the runner's
    /// `Simulation::mark_kinematic_only` — the implementation must
    /// gate the integrator on the carrier so its translational and
    /// rotational state is derived from the parent each tick, not
    /// integrated.
    ///
    /// The default implementation panics with an explicit
    /// "mark_kinematic_only not supported" message so existing
    /// `SimContext` implementors stay source-compatible. Adapters
    /// that own the kinematic-child state machine override this.
    fn mark_kinematic_only(&mut self, child_idx: usize) {
        let _ = child_idx;
        panic!(
            "mark_kinematic_only not supported by this SimContext implementation; \
             provide a SimContext impl that gates the adapter's integrator \
             (e.g. inserts KinematicChildC on the Bevy entity)"
        );
    }

    /// Set body `body_idx`'s root-inertial external force, replacing
    /// any previous value. Mirrors the runner's
    /// `Simulation::set_body_external_force` — invoked from a
    /// `pre_step` closure to schedule time-stamped external load
    /// changes (impulse pairs, on/off pulses) without dropping out of
    /// the recipe path.
    ///
    /// The default implementation panics with an explicit
    /// "set_body_external_force not supported" message so existing
    /// `SimContext` implementors stay source-compatible. Adapters
    /// that own a body-force injection surface (runner's `SimBody`,
    /// the Bevy adapter's `ExternalForceC` component) override this.
    fn set_body_external_force(&mut self, body_idx: usize, force: DVec3) {
        let _ = (body_idx, force);
        panic!(
            "set_body_external_force not supported by this SimContext implementation; \
             provide a SimContext impl that mutates the adapter's external-load \
             surface (e.g. ExternalForceC on the Bevy body entity)"
        );
    }

    /// Set body `body_idx`'s body-frame external torque, replacing any
    /// previous value. Mirrors the runner's
    /// `Simulation::set_body_external_torque` — invoked from a
    /// `pre_step` closure to schedule time-stamped torque changes.
    ///
    /// The default implementation panics with an explicit
    /// "set_body_external_torque not supported" message so existing
    /// `SimContext` implementors stay source-compatible. Adapters
    /// that own a body-torque injection surface override this.
    fn set_body_external_torque(&mut self, body_idx: usize, torque: DVec3) {
        let _ = (body_idx, torque);
        panic!(
            "set_body_external_torque not supported by this SimContext implementation; \
             provide a SimContext impl that mutates the adapter's external-load \
             surface (e.g. ExternalTorqueC on the Bevy body entity)"
        );
    }

    /// Set body `body_idx`'s **structural-frame** external force,
    /// replacing any previous value. Mirrors the runner's
    /// `Simulation::set_body_external_force_struct` — the force is
    /// expressed in the body's structural frame and rotated to
    /// inertial at force-collection time using the body's current
    /// attitude (mirrors JEOD's `dyn_body_collect.cc:219-221`). Use
    /// this entry point for Tier 3 sims that schedule struct-frame
    /// force events (`SIM_verif_attach_detach`'s
    /// `RUN_compute_child_derivative`) so the inertial-frame
    /// contribution tracks the body's attitude across each
    /// integration step.
    ///
    /// The default implementation panics with an explicit
    /// "set_body_external_force_struct not supported" message so
    /// existing `SimContext` implementors stay source-compatible.
    fn set_body_external_force_struct(&mut self, body_idx: usize, force_struct: DVec3) {
        let _ = (body_idx, force_struct);
        panic!(
            "set_body_external_force_struct not supported by this SimContext implementation; \
             provide a SimContext impl that mutates the adapter's structural-frame \
             external-load surface (e.g. ExternalForceStructC on the Bevy body entity)"
        );
    }

    /// Set body `body_idx`'s **structural-frame** external torque,
    /// replacing any previous value. Mirrors the runner's
    /// `Simulation::set_body_external_torque_struct` — the torque is
    /// expressed in the body's structural frame and rotated to the
    /// body frame at force-collection time using the body's
    /// structural-to-body transform.
    ///
    /// The default implementation panics with an explicit
    /// "set_body_external_torque_struct not supported" message so
    /// existing `SimContext` implementors stay source-compatible.
    fn set_body_external_torque_struct(&mut self, body_idx: usize, torque_struct: DVec3) {
        let _ = (body_idx, torque_struct);
        panic!(
            "set_body_external_torque_struct not supported by this SimContext implementation; \
             provide a SimContext impl that mutates the adapter's structural-frame \
             external-load surface (e.g. ExternalTorqueStructC on the Bevy body entity)"
        );
    }

    /// Read body `body_idx`'s current inertial-body left-transformation
    /// quaternion as `glam::DQuat` (xyzw layout). Used by `pre_step`
    /// closures that need to rotate a body-frame load into the inertial
    /// frame before calling [`Self::set_body_external_force`] (whose
    /// argument lives in `RootInertial`). The DQuat is the same value
    /// the integrator reads via `SimBody.rot.q_inertial_body` — convert
    /// with [`astrodyn::JeodQuat::from_glam`] when the closure needs the
    /// scalar-first JEOD layout.
    ///
    /// The default implementation panics with an explicit
    /// "body_q_inertial_body not supported" message so existing
    /// `SimContext` implementors stay source-compatible. Adapters that
    /// expose a body's rotational state override this.
    fn body_q_inertial_body(&self, body_idx: usize) -> DQuat {
        let _ = body_idx;
        panic!(
            "body_q_inertial_body not supported by this SimContext implementation; \
             provide a SimContext impl that reads the adapter's rotational state \
             (e.g. RotationalStateC on the Bevy body entity)"
        );
    }

    /// Attach `body_idx` to a non-body reference frame owned by gravity
    /// source `source_idx`, with a fixed structural-origin `offset`
    /// (parent-frame coordinates, m) and `t_parent_child` rotation
    /// (parent-frame axes → body structural axes). `frame_kind` picks
    /// the source's inertial or planet-fixed frame; mirrors the runner's
    /// [`Simulation::attach_to_frame`](https://docs.rs/astrodyn_runner/latest/astrodyn_runner/simulation/struct.Simulation.html#method.attach_to_frame)
    /// when paired with [`Simulation::source_pfix_frame_id`] /
    /// [`Simulation::source_inertial_frame_id`] for frame lookup.
    ///
    /// After the call, the body's translational + rotational integration
    /// is suppressed and each subsequent step derives state from the
    /// parent frame composed with the captured offset. Used by ref-frame
    /// attach scenarios that schedule attach mid-propagation through
    /// `pre_step`.
    ///
    /// The default implementation panics with an explicit
    /// "attach_to_frame not supported" message so existing `SimContext`
    /// implementors stay source-compatible. Adapters that own a
    /// frame-attach surface (the runner's `Simulation`, the Bevy
    /// adapter's `FrameAttachEvent` bus) override this.
    fn attach_to_frame(
        &mut self,
        body_idx: usize,
        source_idx: usize,
        frame_kind: SourceFrameKind,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        let _ = (body_idx, source_idx, frame_kind, offset, t_parent_child);
        panic!(
            "attach_to_frame not supported by this SimContext implementation; \
             provide a SimContext impl that drives the adapter's frame-attach \
             path (e.g. FrameAttachEvent on the Bevy bus)"
        );
    }

    /// Detach the subtree rooted at `subtree_root` from its current
    /// parent in the mass tree. Mirrors the runner's
    /// [`Simulation::detach_subtree`](https://docs.rs/astrodyn_runner/latest/astrodyn_runner/simulation/struct.Simulation.html#method.detach_subtree)
    /// runtime entry point used by Apollo's staged separation events
    /// (S-IC drop, LM extraction, …): the subtree's composite-body
    /// inertial state is captured at the separation instant, the
    /// parent's composite-CoM-shift is propagated through its
    /// integrated state, and the subtree advances ballistically from
    /// then on until the matching [`Self::attach_subtree_aligned`]
    /// re-merges it. Addressed by `MassBodyId` rather than body index
    /// because the subtree root is typically a tree-only body (no
    /// `SimBody` / no Bevy dynamic entity), e.g. the apollo LM /
    /// service module sub-stages.
    ///
    /// The default implementation panics so existing `SimContext`
    /// implementors stay source-compatible. Adapters that own the
    /// subtree-detach surface (runner's `Simulation::detach_subtree`,
    /// the Bevy adapter's `DetachEvent` against a mass-only entity)
    /// override this.
    fn detach_subtree(&mut self, subtree_root: astrodyn::MassBodyId) {
        let _ = subtree_root;
        panic!(
            "detach_subtree not supported by this SimContext implementation; \
             provide a SimContext impl that drives the adapter's subtree-detach \
             path (e.g. DetachEvent on the Bevy bus against a mass-only entity)"
        );
    }

    /// Re-attach `subtree_root` (previously detached via
    /// [`Self::detach_subtree`]) under `parent` in the mass tree using
    /// named attachment points, running JEOD's
    /// `combine_states_at_attach` momentum-conservation kernel so the
    /// merged composite-body state is bit-identical to the runner's
    /// [`Simulation::attach_subtree_aligned`](https://docs.rs/astrodyn_runner/latest/astrodyn_runner/simulation/struct.Simulation.html#method.attach_subtree_aligned).
    /// The mass-tree must already contain the named mass points on
    /// both bodies (typically declared at scenario-build time); the
    /// adapter looks them up and computes the structural-frame offset
    /// and rotation chain (`mass_attach.cc:103-115` — invert the child
    /// point, apply 180° docking yaw, compose with the parent point)
    /// internally so callers only carry around the symbolic names.
    ///
    /// The default implementation panics so existing `SimContext`
    /// implementors stay source-compatible. Adapters that own the
    /// subtree-attach surface override this.
    fn attach_subtree_aligned(
        &mut self,
        subtree_root: astrodyn::MassBodyId,
        subtree_point: &str,
        parent: astrodyn::MassBodyId,
        parent_point: &str,
    ) {
        let _ = (subtree_root, subtree_point, parent, parent_point);
        panic!(
            "attach_subtree_aligned not supported by this SimContext implementation; \
             provide a SimContext impl that drives the adapter's subtree-attach \
             path (e.g. AttachEvent against a mass-only entity with offset / \
             rotation derived from named mass points)"
        );
    }

    /// Set the simulation's `time_scale_factor` on the underlying
    /// `SimulationTime`. Mirrors the runner's
    /// `sim.time.time_scale_factor = factor` field write. Used by
    /// `pre_step` closures that schedule a time-direction flip (the
    /// SIM_7_time_reversal pattern: forward propagation until the
    /// reversal instant, then `factor = -1.0` so the dynamic time
    /// scales TAI / TT / TDB / GMST reverse while `simtime` keeps
    /// advancing monotonically).
    ///
    /// The default implementation panics with an explicit
    /// "set_time_scale_factor not supported" message so existing
    /// `SimContext` implementors stay source-compatible. Adapters
    /// that own a `SimulationTime` surface (the runner's
    /// `Simulation::time`, the Bevy adapter's `SimulationTimeR`
    /// resource) override this.
    fn set_time_scale_factor(&mut self, factor: f64) {
        let _ = factor;
        panic!(
            "set_time_scale_factor not supported by this SimContext implementation; \
             provide a SimContext impl that writes the adapter's SimulationTime \
             scale factor (e.g. `SimulationTimeR.0.time_scale_factor = factor` \
             on the Bevy resource)"
        );
    }
}

/// Which of a gravity source's reference frames to attach to.
///
/// Pairs with [`SimContext::attach_to_frame`] to pick between the
/// non-rotating inertial frame and the rotating planet-fixed frame.
/// Adapter-neutral so the runner-side impl can resolve to a
/// `astrodyn_runner::FrameId` and the Bevy-side impl can resolve to the
/// source entity's `FrameEntityC` / `PfixFrameEntityC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFrameKind {
    /// The source's inertial frame (`source_inertial_frame_id` on the
    /// runner; `FrameEntityC` on the Bevy source entity).
    Inertial,
    /// The source's planet-fixed (rotating) frame
    /// (`source_pfix_frame_id` on the runner; `PfixFrameEntityC` on the
    /// Bevy source entity). Requires the source to have a rotation
    /// model — the underlying adapter call panics if the source lacks
    /// a pfix frame.
    Pfix,
}

/// Closure type produced by a [`PreStepBuilder`]. Invoked before the
/// simulation advances; the calling cadence is recipe-configurable via
/// [`PreStepCadence`] on the owning [`VerificationCase`].
///
/// The `time` argument is the simulation time *at the end of the
/// upcoming interval* in seconds since the simulation epoch. Under
/// [`PreStepCadence::PerRecord`] that interval spans the gap to the
/// next reference-CSV checkpoint; under [`PreStepCadence::PerTick`]
/// it is a single integrator `dt`.
///
/// Most closures (third-body ephemeris updates, tide-host source
/// positions) naturally line up with the reference-CSV cadence: JEOD's
/// Trick scheduler invokes those at the record rate, so a per-record
/// closure here matches bit-for-bit. Closures whose physical decision
/// can flip between two reference-CSV records — scheduled external
/// force / torque pulses, attach/detach events — must opt into
/// [`PreStepCadence::PerTick`] so the on/off boundary lines up with
/// JEOD's per-tick scheduler. The exception is recipes whose
/// reference cadence already equals the integrator `dt` (typically
/// [`CsvReference::SyntheticTimes`] scenarios such as
/// `sim_attach_detach_trajectory::simple`): there every record *is* a
/// tick, so [`PreStepCadence::PerRecord`] and [`PreStepCadence::PerTick`]
/// produce identical call sequences and the per-record variant is the
/// conventional choice.
///
/// Closures that need a TDB Julian date should derive it as
/// `j2000_jd + time / 86_400.0` (assuming a J2000 epoch), or capture
/// the epoch's JD when they're constructed by their
/// [`PreStepBuilder`].
pub type PreStepClosure = Box<dyn FnMut(&mut dyn SimContext, f64) + Send>;

/// Calling cadence for a [`PreStepClosure`].
///
/// JEOD's Trick scheduler invokes its hooks at two distinct rates: the
/// reference-CSV record rate (typically 1–60 s) for third-body /
/// ephemeris-driven source updates and for tide-host source positions,
/// and the dynamics rate (e.g. 32 Hz / dt = 0.03125 s for SIM_dyncomp)
/// for scheduled external force / torque pulses. Our `pre_step` slot
/// must match whichever cadence the JEOD recipe pairs with, or the
/// runner-vs-JEOD comparison drifts:
///
/// - Calling a per-record closure per tick re-evaluates ephemeris at a
///   much finer cadence than JEOD does, perturbing the differential
///   third-body acceleration.
/// - Calling a per-tick closure per record freezes the closure's
///   decision across the entire interval, missing the on/off boundary
///   the recipe schedules.
///
/// The variant is recipe-configurable on [`VerificationCase`] rather
/// than global so each recipe pairs with its own JEOD cadence
/// independently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreStepCadence {
    /// Invoke the closure once before each `sim.step_until(record.time)`
    /// call, i.e. once per reference-CSV checkpoint. Matches JEOD's
    /// per-record scheduler invocation for third-body / ephemeris
    /// updates and tide-host source positions.
    #[default]
    PerRecord,
    /// Invoke the closure before every integration tick (each
    /// `sim.step()` / `App::update()`). Matches JEOD's per-tick
    /// scheduler invocation for scheduled external force / torque
    /// pulses whose on/off boundary lives between record cadences.
    PerTick,
}

/// Factory for a [`PreStepClosure`]. Invoked once at the start of
/// `run_and_assert` with the t=0 [`InitialConditions`], so the closure
/// it returns can capture state (a loaded ephemeris, J2000 JD, source
/// indices, …) that the per-step body would otherwise re-derive on every
/// call.
pub type PreStepBuilder = fn(&InitialConditions) -> PreStepClosure;

/// Initial conditions extracted from the t=0 row of a reference CSV and
/// passed to a scenario constructor by `run_and_assert`. This lets the
/// runner parse each reference CSV exactly once: it loads the full
/// trajectory, hands the t=0 record here to build the scenario, and
/// reuses the rest of the trajectory for the per-step comparison.
///
/// All variants use raw `glam` types so this struct stays adapter-
/// neutral (no dependency on `astrodyn_verif_jeod` from `astrodyn` outside
/// of dev-deps).
///
/// **Quaternion convention.** `glam::DQuat` is laid out as `(x, y, z, w)`
/// where `w` is the scalar component. JEOD's convention is scalar-first
/// `[q0, q1, q2, q3]` where `q0` is the scalar. A JEOD quaternion
/// `[q0, q1, q2, q3]` therefore maps to
/// `DQuat { x: q1, y: q2, z: q3, w: q0 }`. Scenarios that need a
/// [`astrodyn::JeodQuat`] convert via `JeodQuat::from_glam`.
#[derive(Clone, Debug, Default)]
pub struct InitialConditions {
    /// Reference time (seconds since the sim epoch). Always populated.
    pub time: f64,
    /// RootInertial position. Always populated for the variants used by
    /// migrated Tier 3 cases.
    pub position: DVec3,
    /// RootInertial velocity. Always populated for the variants used by
    /// migrated Tier 3 cases.
    pub velocity: DVec3,
    /// Body-frame attitude quaternion in `glam::DQuat` layout
    /// `(x, y, z, w)` where `w` is the scalar. JEOD's scalar-first
    /// convention `[q0, q1, q2, q3]` (with `q0` scalar) maps to
    /// `DQuat { x: q1, y: q2, z: q3, w: q0 }`. `Some` for 6-DOF cases,
    /// `None` for 3-DOF (point-mass translational-only) scenarios.
    pub quaternion: Option<DQuat>,
    /// Body-frame angular velocity. `Some` for 6-DOF cases, `None` for
    /// 3-DOF.
    pub ang_vel: Option<DVec3>,
}

/// A reference-CSV file used by a Tier 3 verification case.
///
/// Each variant tags a distinct column layout produced by one of JEOD's
/// `log_state_ASCII` configurations. The wrapped `&'static str` is the
/// file name relative to the workspace `test_data/` directory. The
/// runner-side `run_and_assert` machinery dispatches on the variant to
/// pick the right loader.
#[derive(Clone, Debug)]
pub enum CsvReference {
    /// 80-column SIM_dyncomp state CSV consumed as a 3-DOF reference:
    /// position/velocity only — quaternion and ang_vel columns are
    /// dropped at the `crate::crossval::StateLog` layer. Use
    /// this for scenarios that build a `astrodyn::VehicleConfig` without `rot`,
    /// so per-step compares don't synthesize spurious rotational
    /// reference values from CSV columns the simulation never produces.
    Dyncomp3Dof(&'static str),
    /// 80-column SIM_dyncomp state CSV consumed as a 6-DOF reference:
    /// position/velocity *plus* `composite_body.quaternion` and
    /// `composite_body.ang_vel` are populated on the reference
    /// `crate::crossval::StateLog`.
    Dyncomp6Dof(&'static str),
    /// 21+-column SIM_OrbElem CSV (classical elements + state).
    Orbelem(&'static str),
    /// 17+-column SIM_LVLH CSV (T_parent_this + ang_vel_mag + state).
    Lvlh(&'static str),
    /// 16+-column SIM_NED CSV (geodetic + spherical altitudes/lat/lon
    /// + state).
    Ned(&'static str),
    /// 7-column SIM_3_ORBIT SRP CSV (time + pos + vel).
    Srp(&'static str),
    /// 9-column SIM_1_BASIC SRP CSV (force, torque, flux, temperature).
    SrpBasic(&'static str),
    /// 11-column SIM_VER_DRAG CSV (aero force/torque + inertial vel +
    /// accel mag).
    Drag(&'static str),
    /// 56-column SIM_Euler CSV (36 angles + state + T + quat).
    Euler(&'static str),
    /// 8-column SIM_SolarBeta CSV (time + beta + interleaved pos/vel).
    SolarBeta(&'static str),
    /// 11-column SIM_2A_SHADOW_CALC CSV.
    Shadow(&'static str),
    /// 26-column SIM_torque_compare_simple CSV.
    TorqueSimple(&'static str),
    /// 9-column atmosphere-trajectory CSV (state + density + temp).
    AtmosTraj(&'static str),
    /// 14-column aero-trajectory CSV (state + aero force/torque + density).
    AeroTraj(&'static str),
    /// 7-column trajectory CSV with schema `time + pos[3] + vel[3]`.
    /// Used by any sim whose `log_state_ASCII` config emits exactly the
    /// composite-body inertial state (no rotation matrix, quaternion, or
    /// angular velocity columns). Originating sims include `SIM_orbinit`,
    /// `SIM_GJ_test`, and `SIM_Planetary` — the variant is generic over
    /// the schema, not specific to any one of them.
    OrbInit(&'static str),
    /// 8-column SIM_tide_verif CSV (time + pos + vel + dC20).
    Tide(&'static str),
    /// 14-column SIM_ref_attach state CSV (time + pos + vel + q + ang_vel).
    /// JEOD's SIM_ref_attach `IntegLoop` runs at `DYNAMICS = 1.0` s but
    /// Trick logs at 0.5 s; the half-second rows simply repeat the
    /// previous integer-second integrator output, so the loader drops
    /// them. Pairs with [`Self::file_name`] for the filename; `dt`
    /// names the integrator cadence the half-second filter quantizes
    /// against.
    RefAttach {
        /// CSV file name under `test_data/`.
        file: &'static str,
        /// Integrator timestep in seconds. Half-second rows that don't
        /// land on an integer multiple of `dt` are dropped at load
        /// time so the per-step comparison cadence stays aligned with
        /// the integration cadence.
        dt: f64,
    },
    /// 57-column SIM_Relative two-body CSV (time + interleaved vehA
    /// state[25] + interleaved vehB state[25] + JEOD-logged relative
    /// translational state[6]). Used by the runner-vs-JEOD oracle
    /// (`tier3_sim_relative.rs`) to assert
    /// [`astrodyn::compute_relative_state`] against JEOD's own
    /// SIM_Relative output via [`ExtrasComparator::Relative`].
    Relative(&'static str),
    /// CSV consumed for time-cadence only — the per-variant loaders
    /// don't know how to parse the body of this file (or it carries
    /// columns the cross-validation report would misinterpret), so the
    /// dispatcher reads only column 0 (`sys.exec.out.time {s}`) and
    /// emits [`crate::crossval::StateLog`]s with `position` /
    /// `velocity` left as `None`.
    ///
    /// Use this when a recipe needs to step at JEOD's reference
    /// cadence but the assertion is parity-only (runner ↔ bevy
    /// bit-identity through
    /// [`astrodyn_verif_parity::VerificationCaseParityExt::run_and_assert_parity`])
    /// rather than tolerance-bounded against JEOD-logged state. The
    /// matching tier3 sibling can keep using a hand-rolled loader for
    /// its own column layout — only the parity trait routes through
    /// this variant.
    ///
    /// [`astrodyn_verif_parity::VerificationCaseParityExt::run_and_assert_parity`]: https://github.com/simnaut/astrodyn/blob/main/crates/astrodyn_verif_parity/src/lib.rs
    TimesOnly(&'static str),
    /// Synthetic time cadence — no CSV file on disk. Emits
    /// `num_steps + 1` records at times `0, dt, 2·dt, …, num_steps·dt`.
    /// Used by parity-only recipes that have no JEOD reference
    /// trajectory but still need a checkpoint cadence to drive
    /// [`astrodyn_verif_parity::VerificationCaseParityExt::run_and_assert_parity`].
    ///
    /// The runner-side
    /// [`crate::run_verification::VerificationCaseExt::run_and_assert`]
    /// also accepts this variant — it generates the same times in
    /// memory and runs the propagation loop with `position` /
    /// `velocity` left as `None` on every record (same shape as
    /// [`Self::TimesOnly`]). Recipes that pair with this variant
    /// must use all-zero tolerances so the runner-vs-JEOD comparison
    /// opts out of every assertion (the documented "all-zero skips
    /// the metric group" rule).
    ///
    /// This is the on-disk-CSV-free sibling of
    /// [`Self::TimesOnly`]. Prefer it when the recipe doesn't pair
    /// with a JEOD-generated trajectory at all (purely synthetic
    /// scenarios — `bevy_parity_kinematic_propagation`,
    /// `bevy_parity_attach_detach_trajectory`, the Bevy-mechanism
    /// SRP family, etc.); it eliminates the
    /// committed-but-otherwise-unused CSV fixture and the
    /// `test_data_path` lookup that comes with it.
    SyntheticTimes {
        /// Step size between consecutive checkpoints, in seconds.
        dt: f64,
        /// Number of `dt`-sized intervals; the dispatch emits
        /// `num_steps + 1` records (the t=0 row plus `num_steps`
        /// stepped records).
        num_steps: usize,
    },
}

impl CsvReference {
    /// Returns the underlying file name (relative to `test_data/`),
    /// or `None` for the synthetic-cadence variant.
    pub fn file_name(&self) -> Option<&'static str> {
        match self {
            CsvReference::Dyncomp3Dof(s)
            | CsvReference::Dyncomp6Dof(s)
            | CsvReference::Orbelem(s)
            | CsvReference::Lvlh(s)
            | CsvReference::Ned(s)
            | CsvReference::Srp(s)
            | CsvReference::SrpBasic(s)
            | CsvReference::Drag(s)
            | CsvReference::Euler(s)
            | CsvReference::SolarBeta(s)
            | CsvReference::Shadow(s)
            | CsvReference::TorqueSimple(s)
            | CsvReference::AtmosTraj(s)
            | CsvReference::AeroTraj(s)
            | CsvReference::OrbInit(s)
            | CsvReference::Tide(s)
            | CsvReference::Relative(s)
            | CsvReference::TimesOnly(s) => Some(s),
            CsvReference::RefAttach { file, .. } => Some(file),
            CsvReference::SyntheticTimes { .. } => None,
        }
    }
}

/// Per-component tolerances for trajectory cross-validation.
///
/// Each field corresponds to a `CrossvalReport::assert_*` method —
/// `position_m` per axis, `velocity_m_s` per axis, scalar
/// `quat_angle_rad`, `ang_vel_rad_s` per axis. `extras` lets a Tier 3
/// case attach scenario-specific tolerances (e.g., the GR perihelion-
/// advance arc-second-per-century check on the Mercury case).
///
/// **Skip semantics.** A whole metric group is skipped only when *all*
/// of its component tolerances are zero (`position_m: [0.0; 3]`,
/// `velocity_m_s: [0.0; 3]`, scalar `quat_angle_rad == 0.0`,
/// `ang_vel_rad_s: [0.0; 3]`). This is the pattern used by 3-DOF cases
/// to opt out of rotational assertions. A non-zero entry alongside a
/// zero entry in the same array does *not* skip the zero axis — the
/// runner still asserts `error_axis < 0.0` on it, which always fails.
/// Mixing zero and non-zero entries within a single array is therefore
/// almost always a configuration mistake.
#[derive(Clone, Debug)]
pub struct Tolerances {
    /// Per-axis position tolerance (m). All-zero opts out of the
    /// position assertion entirely.
    pub position_m: [f64; 3],
    /// Per-axis velocity tolerance (m/s). All-zero opts out of the
    /// velocity assertion entirely.
    pub velocity_m_s: [f64; 3],
    /// Scalar quaternion-angle tolerance (rad). Zero opts out of the
    /// attitude assertion entirely.
    pub quat_angle_rad: f64,
    /// Per-axis angular-velocity tolerance (rad/s). All-zero opts out of
    /// the angular-velocity assertion entirely.
    pub ang_vel_rad_s: [f64; 3],
    /// Family-specific extras: `(name, abs-tolerance)` pairs that the
    /// runner asserts against `report.add_extra(name, ...)` outputs.
    pub extras: &'static [(&'static str, f64)],
}

/// Per-family extras comparator dispatched by `run_and_assert`.
///
/// Each variant tags a family-specific extractor that pairs a
/// [`crate::verification::CsvReference`]'s typed record at
/// step *k* with the runner-side `astrodyn_runner::VehicleOutput` at the
/// same step, yielding `(name, abs_error)` pairs the runner
/// accumulates as max errors and asserts against
/// [`Tolerances::extras`].
///
/// The runner-side dispatch lives in `crate::run_verification`
/// (it has access to typed records and `VehicleOutput`); this enum is
/// adapter-neutral so `VerificationCase` itself stays in `astrodyn`.
#[derive(Clone, Debug)]
pub enum ExtrasComparator {
    /// Classical orbital elements: 7 extras (sma, eccentricity, inclination,
    /// arg_periapsis, long_asc_node, true_anom, mean_anom).
    Orbelem,
    /// LVLH frame: 2 extras (`t_parent_this` matrix-element max error,
    /// `ang_vel` magnitude error).
    Lvlh,
    /// Geodetic state: 3 extras (`altitude`, `latitude`, `longitude`).
    /// `spherical=true` compares against the spherical-Earth columns;
    /// `false` (default) compares against ellipsoidal columns.
    Ned {
        /// `true` compares against the spherical-Earth NED columns;
        /// `false` (default) compares against the ellipsoidal columns.
        spherical: bool,
    },
    /// Euler angles: 3 extras (`euler_roll`, `euler_pitch`, `euler_yaw`)
    /// computed against JEOD's logged quaternion via our own port of the
    /// Euler-from-matrix conversion (self-consistency check of our Euler
    /// extractor against the JEOD-quaternion reference).
    Euler,
    /// Same Euler self-consistency check as [`Self::Euler`] but reading
    /// the reference quaternion from a [`CsvReference::Dyncomp6Dof`]
    /// `composite_body.quaternion` row rather than a SIM_Euler CSV. Used
    /// by SIM_Euler runs that drive themselves from the SIM_dyncomp
    /// RUN_2 trajectory.
    DyncompEuler,
    /// Solar beta angle: 1 extra (`beta`) comparing `body.solar_beta`
    /// against the matching column in JEOD's SIM_SolarBeta reference
    /// CSV. Pairs with [`CsvReference::SolarBeta`]. Solar beta in this
    /// codebase is constrained to `[-π/2, π/2]` per
    /// `astrodyn_math::solar_beta_angle_*`, so the metric is a plain
    /// absolute difference (no angular wrap-around to handle).
    SolarBeta,
    /// Solid-body tidal ΔC20: 1 extra (`dc20`) comparing the
    /// simulation's per-step ΔC20 (sourced from
    /// `Simulation::source_delta_c20(earth_source_idx)`) against the
    /// `dC20` column logged by JEOD's SIM_tide_verif. Pairs with
    /// [`CsvReference::Tide`]. The recipe carries the Earth source
    /// index because dC20 is per-source, not per-body.
    TideDc20 {
        /// Index (in the simulation's source table) of the Earth source
        /// whose ΔC20 series the comparator will sample.
        earth_source_idx: usize,
    },
    /// SIM_Relative two-body relative state: 2 extras (`rel_pos`,
    /// `rel_vel`) computed via [`astrodyn::compute_relative_state`] on
    /// the runner's bodies 0 and 1 and compared against the
    /// JEOD-logged relative position / velocity vectors in CSV
    /// columns 51–56. The metric is a vector-magnitude error
    /// (`(ours - reference).length()`), matching the bespoke
    /// `tier3_sim_relative.rs` assertion shape exactly.
    Relative,
}

/// A single Tier 3 verification case.
///
/// Phase 6 shipped the type; Phase 7+ populates `verification::*`
/// constructors that produce one of these per existing Tier 3 test.
/// `run_and_assert` is provided by `crate::run_verification`
/// because materializing the scenario into a runtime
/// `astrodyn_runner::Simulation` is runner-specific.
#[derive(Clone, Debug)]
pub struct VerificationCase {
    /// Unique name used for `target/tier3_crossval/{name}.json` reports.
    pub name: &'static str,
    /// Scenario constructor. Receives the t=0 [`InitialConditions`]
    /// extracted from `reference` so the scenario does not need to
    /// re-parse the reference CSV. The fn pointer stays adapter-neutral
    /// so the runner and (Phase 9) Bevy adapter consume the same
    /// scenario.
    pub scenario: fn(&InitialConditions) -> SimulationBuilder,
    /// Reference CSV produced by the corresponding JEOD verification
    /// simulation.
    pub reference: CsvReference,
    /// Total propagation duration. The runner truncates iteration over
    /// the reference CSV to records with `record.time <= duration`.
    /// `Time::new::<second>(0.0)` (or any value `<= 0.0`) means *use the
    /// full CSV*. If `duration` exceeds the last record's time the loop
    /// simply runs to the end (no extrapolation).
    pub duration: Time,
    /// Per-component tolerances for the cross-validation report.
    pub tolerances: Tolerances,
    /// Optional per-family extras comparator. When `Some`, the runner
    /// computes the family's `(name, error)` pairs alongside the state
    /// log and asserts each against the matching entry in
    /// [`Tolerances::extras`].
    pub extras: Option<ExtrasComparator>,
    /// Optional pre-step hook factory paired with its calling cadence.
    ///
    /// When `Some((builder, cadence))`, the runner calls `builder` once
    /// at the start of `run_and_assert` (with the t=0
    /// [`InitialConditions`]) to obtain a [`PreStepClosure`], then
    /// invokes that closure at the given cadence:
    ///
    /// - [`PreStepCadence::PerRecord`] — once before each
    ///   `sim.step_until(record.time)` call (the JEOD-scheduler match
    ///   for third-body / ephemeris / tide / SRP source-position
    ///   updates).
    /// - [`PreStepCadence::PerTick`] — once before each `sim.step()` /
    ///   `App::update()` tick (the JEOD-scheduler match for scheduled
    ///   external force / torque pulses whose on/off boundary flips
    ///   between record cadences).
    ///
    /// Use this to inject mid-flight state — source ephemeris updates,
    /// scheduled external force/torque pulses, or runtime mass-tree
    /// changes — at whichever rate JEOD itself drives.
    ///
    /// The factory pattern lets the closure capture run-once state (a
    /// loaded DE421 ephemeris, J2000 JD, source indices) that the
    /// per-call body would otherwise re-derive on every invocation.
    pub pre_step: Option<(PreStepBuilder, PreStepCadence)>,
}
