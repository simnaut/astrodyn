//! Per-body derived state functions.
//!
//! Post-integration observational quantities computed from the integrated state.
//! Each function is pure (no side effects) and takes explicit parameters so that
//! any ECS adapter can call it from a system function.
//!
//! JEOD_INV: TS.01 — dynamic-registry-erased return-type boundary. The
//! `<SelfPlanet>`-tagged returns from [`compute_orbital_elements`] (and
//! the `<SelfRef, SelfRef>` instantiations of [`compute_relative_state`]
//! / [`compute_lvlh_relative_state`] used by the standalone runner)
//! exist because the runner's per-body planet identity is keyed by a
//! dynamic source index rather than a compile-time `<P>`; the typed
//! `_typed` siblings in this module are the compile-time-pinned API
//! surface mission code uses. The wildcard tags here are the documented
//! storage / erasure boundary for that runtime-resolved choice. The
//! lint at `tests/self_ref_self_planet_discipline.rs` enforces this
//! globally.

use glam::{DQuat, DVec3};

use crate::{EulerSequence, LvlhFrame, OrbitalElements, RotationalState};
use astrodyn_math::OrbitalError;
use astrodyn_quantities::aliases::{Position, Velocity};
use astrodyn_quantities::dims::GravParam;
use astrodyn_quantities::ext::Vec3Ext;
use astrodyn_quantities::frame::{BodyFrame, Lvlh, RootInertial, SelfPlanet, Vehicle};
use uom::si::angle::radian;
use uom::si::f64::Angle;

/// Relative state between two bodies.
///
/// Position/velocity of `subject` relative to `reference` live on
/// [`Self::trans`], whose [`RelativeTranslation`] sum type encodes the
/// runtime-conditional frame choice in the type system: when the
/// reference carries a [`RotationalState`] the values are rotated into
/// the reference body frame ([`RelativeTranslation::BodyFrame`],
/// matching JEOD convention `S_{ref:subj}`); when the reference has no
/// rotational state they remain in the inertial frame
/// ([`RelativeTranslation::Inertial`]). A single compile-time phantom
/// cannot losslessly cover both branches, so the variant tag — set by
/// the producer based on whether `ref_rot` is `Some`/`None` — is what
/// downstream consumers pattern-match on to retrieve typed
/// `Position<F>` / `Velocity<F>` values for the right frame.
///
/// The angular kinematics are typed: [`Self::ang_vel`] is in the
/// subject body frame. The quaternion is the relative attitude
/// (reference-to-subject) and is stored as a `DQuat` for convenience;
/// the convention matches JEOD's left-multiplication
/// reference-body-to-subject-body rotation.
///
/// # Vehicle phantoms
///
/// `RelativeState` carries two independent vehicle identities:
///
/// - `Subject: Vehicle` names the vehicle whose state is being
///   described; the typed [`Self::ang_vel`] sits in that vehicle's
///   body frame.
/// - `Reference: Vehicle` names the vehicle the state is expressed
///   relative to; the body-frame variant of [`RelativeTranslation`]
///   sits in that vehicle's body frame.
///
/// Mission code that pins both identities at compile time (via
/// [`define_vehicle!`](astrodyn_quantities::define_vehicle)) gets a
/// compile-time guard against pair confusion — a
/// `RelativeState<Iss, Soyuz>` cannot be passed to a consumer
/// expecting `RelativeState<Iss, Cygnus>`. The Bevy adapter and
/// standalone runner instantiate `RelativeState<SelfRef, SelfRef>`
/// because their per-entity storage decides the subject and reference
/// identities at runtime.
#[derive(Debug, Clone)]
pub struct RelativeState<Subject: Vehicle, Reference: Vehicle> {
    /// Translational state of subject relative to reference. The
    /// variant identifies the frame the values live in — see
    /// [`RelativeTranslation`] for the branch contract. The
    /// `Reference` phantom propagates into the body-frame variant so
    /// a `RelativeState<_, Iss>` cannot accidentally feed a consumer
    /// expecting `RelativeTranslation<Soyuz>`.
    pub trans: RelativeTranslation<Reference>,
    /// Relative quaternion: reference body frame → subject body frame.
    pub quaternion: DQuat,
    /// Angular velocity of subject relative to reference, in the
    /// subject body frame. The `<Subject>` phantom names the subject
    /// vehicle so a consumer that holds a typed
    /// `AngularVelocity<BodyFrame<Iss>>` cannot accidentally consume
    /// an `AngularVelocity<BodyFrame<Soyuz>>`; the Bevy adapter and
    /// runner instantiate `<SelfRef>` because their per-entity
    /// storage decides the subject identity at runtime.
    pub ang_vel: astrodyn_quantities::aliases::AngularVelocity<BodyFrame<Subject>>,
}

impl<Subject: Vehicle, Reference: Vehicle> RelativeState<Subject, Reference> {
    /// Type-level witness that this relative state carries the caller's
    /// expected `(Subject, Reference)` vehicle phantoms. Compiles only
    /// when both phantoms agree; on mismatch the
    /// [`astrodyn_quantities::diagnostics::CompatibleVehiclePair`] bound
    /// fails and surfaces a physics-language diagnostic naming both
    /// expected and found pairs instead of a `PhantomData<…>` wall.
    ///
    /// The method is a no-op (returns `self`) and has zero runtime
    /// cost; it exists to thread the diagnostic into call sites that
    /// would otherwise see only a generic type-mismatch error.
    ///
    /// # Compile-time mismatch
    ///
    /// ```compile_fail
    /// use astrodyn_quantities::define_vehicle;
    /// use astrodyn::{compute_relative_state, RelativeState, RotationalState, TranslationalState};
    /// use glam::DVec3;
    ///
    /// define_vehicle!(Iss);
    /// define_vehicle!(Soyuz);
    /// define_vehicle!(Cygnus);
    ///
    /// let trans = TranslationalState { position: DVec3::ZERO, velocity: DVec3::ZERO };
    /// let rel: RelativeState<Iss, Soyuz> =
    ///     compute_relative_state::<Iss, Soyuz>(&trans, None, &trans, None);
    /// // Asserting the wrong reference vehicle fires the
    /// // `CompatibleVehiclePair` diagnostic.
    /// let _ = rel.assert_pair::<Iss, Cygnus>();
    /// ```
    #[inline]
    pub fn assert_pair<S: Vehicle, R: Vehicle>(self) -> Self
    where
        (): astrodyn_quantities::diagnostics::CompatibleVehiclePair<Subject, Reference, S, R>,
    {
        self
    }
}

/// Frame-tagged translational state inside a [`RelativeState`].
///
/// The [`compute_relative_state`] producer chooses the variant based on
/// the runtime presence of the reference's rotational state — JEOD's
/// `decr_left` convention rotates into the reference body frame iff a
/// rotational state is available, and otherwise leaves the difference
/// in the inertial frame. Encoding the choice as a sum-typed variant
/// lets each variant carry a distinct phantom-tagged
/// [`Position`]/[`Velocity`] pair, so a consumer that mixes a
/// body-frame value with a root-inertial value is a compile error.
///
/// # Vehicle phantom
///
/// `RelativeTranslation` carries a single `<Reference: Vehicle>`
/// parameter naming the *reference* vehicle in whose body frame the
/// `BodyFrame` variant's values are expressed. The `Inertial` variant
/// is in the root inertial frame and does not depend on a vehicle
/// identity, but the parameter is still threaded through the type so
/// the two variants share a consistent compile-time pair shape with
/// [`RelativeState`]'s `<Subject, Reference>`. Mission code that pins
/// the reference at compile time gets a cross-pair guard:
/// `RelativeTranslation<Iss>` and `RelativeTranslation<Soyuz>` are
/// distinct types. The Bevy adapter and standalone runner instantiate
/// `<Reference = SelfRef>`.
///
/// Pattern-match to read:
///
/// ```ignore
/// match rel.trans {
///     RelativeTranslation::BodyFrame { position, velocity } => { /* typed body-frame */ }
///     RelativeTranslation::Inertial  { position, velocity } => { /* typed root-inertial */ }
/// }
/// ```
///
/// For consumers that just want the raw `DVec3` (e.g. CSV-reference
/// distance metrics in Tier 3 tests where both branches are valid
/// inputs to the same scalar `length()`), [`Self::position_raw`] and
/// [`Self::velocity_raw`] return the underlying SI vector without
/// committing to a phantom — they're the deliberate escape hatch for
/// branch-agnostic numerical consumers.
///
/// # Compile-time frame guard
///
/// The body-frame variant carries a [`Position<BodyFrame<Reference>>`]
/// while the inertial variant carries a [`Position<RootInertial>`].
/// Adding (or otherwise mixing without a deliberate `FrameTransform`)
/// the two phantoms is a compile error. The doctest below documents
/// the guard and is consumed by `cargo test --doc` as a `compile_fail`
/// rustdoc test; if the body-frame and inertial-frame `Position`
/// types ever became compatible, this test would start passing and
/// fail the doctest.
///
/// ```compile_fail
/// use astrodyn_quantities::ext::Vec3Ext;
/// use astrodyn_quantities::frame::{BodyFrame, RootInertial, SelfRef};
/// use glam::DVec3;
///
/// let body_frame_pos = DVec3::ZERO.m_at::<BodyFrame<SelfRef>>();
/// let inertial_pos   = DVec3::ZERO.m_at::<RootInertial>();
/// // The two phantoms are kind-distinct — addition requires either
/// // matching frames or an explicit `FrameTransform`. The compiler
/// // refuses this expression at the type level:
/// let _ = body_frame_pos + inertial_pos;
/// ```
///
/// # Compile-time cross-pair guard
///
/// The reference-vehicle phantom makes a `BodyFrame<Iss>` value
/// kind-distinct from a `BodyFrame<Soyuz>` value. The doctest below
/// shows that a consumer holding a `Position<BodyFrame<Iss>>` cannot
/// accidentally bind it from a `RelativeTranslation<Soyuz>`'s
/// body-frame variant — the destructure refuses to typecheck.
///
/// ```compile_fail
/// use astrodyn_quantities::aliases::Position;
/// use astrodyn_quantities::define_vehicle;
/// use astrodyn_quantities::frame::BodyFrame;
/// use astrodyn::RelativeTranslation;
///
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
///
/// fn binder(t: RelativeTranslation<Soyuz>) {
///     // Would silently accept Soyuz-frame data into the Iss slot if
///     // the phantom didn't propagate through the variant — the
///     // compiler refuses this binding:
///     let RelativeTranslation::BodyFrame { position, .. } = t else { return; };
///     let _bound: Position<BodyFrame<Iss>> = position;
/// }
/// ```
#[derive(Debug, Clone)]
pub enum RelativeTranslation<Reference: Vehicle> {
    /// Reference rotational state was present: position/velocity are
    /// rotated into the reference body frame (JEOD `S_{ref:subj}`).
    /// The `BodyFrame<Reference>` phantom names the reference
    /// vehicle's body frame at the type level — distinct from the
    /// subject's body frame on [`RelativeState::ang_vel`].
    BodyFrame {
        /// Position of subject relative to reference, expressed in the
        /// reference body frame.
        position: Position<BodyFrame<Reference>>,
        /// Velocity of subject relative to reference, expressed in the
        /// reference body frame, with the JEOD Coriolis correction
        /// applied (`v - ω_ref × r`).
        velocity: Velocity<BodyFrame<Reference>>,
    },
    /// Reference rotational state was absent: position/velocity are
    /// the raw inertial-frame difference. The `RootInertial` phantom
    /// is the simulation's root inertial frame — for body-state
    /// inputs read from an integration-frame storage slot the
    /// convention rests on the call site (the producer cannot know
    /// statically that the inputs are root-inertial vs. some
    /// `PlanetInertial<P>` integration frame). Mission code that
    /// holds typed inputs should use the typed sibling API path
    /// rather than feeding raw `DVec3` here.
    Inertial {
        /// Position of subject relative to reference, in the inertial
        /// frame.
        position: Position<RootInertial>,
        /// Velocity of subject relative to reference, in the inertial
        /// frame.
        velocity: Velocity<RootInertial>,
    },
}

impl<Reference: Vehicle> RelativeTranslation<Reference> {
    /// Raw position vector in whichever frame the variant carries.
    ///
    /// Provided for branch-agnostic numerical consumers (e.g. scalar
    /// distance metrics). Frame-aware consumers should pattern-match
    /// on the variant to obtain the typed [`Position<F>`] and route
    /// it through the typed boundary.
    pub fn position_raw(&self) -> DVec3 {
        match self {
            Self::BodyFrame { position, .. } => position.raw_si(),
            Self::Inertial { position, .. } => position.raw_si(),
        }
    }

    /// Raw velocity vector in whichever frame the variant carries.
    /// Companion to [`Self::position_raw`].
    pub fn velocity_raw(&self) -> DVec3 {
        match self {
            Self::BodyFrame { velocity, .. } => velocity.raw_si(),
            Self::Inertial { velocity, .. } => velocity.raw_si(),
        }
    }

    /// Type-level witness that this translation carries the caller's
    /// expected reference-vehicle phantom `R`. Compiles only when
    /// `Reference == R`; on mismatch the
    /// [`astrodyn_quantities::diagnostics::CompatibleVehicles`] bound fails
    /// and surfaces a physics-language diagnostic naming the expected
    /// and found vehicles instead of a `PhantomData<…>` wall.
    ///
    /// The method is a no-op (returns `self`) and has zero runtime
    /// cost; it exists so a consumer holding a
    /// `RelativeTranslation<Iss>` can refuse — at compile time, with a
    /// physics-language message — a value built for a different
    /// reference vehicle.
    ///
    /// # Compile-time mismatch
    ///
    /// ```compile_fail
    /// use glam::DVec3;
    /// use astrodyn_quantities::define_vehicle;
    /// use astrodyn::{compute_relative_state, RelativeTranslation, TranslationalState};
    ///
    /// define_vehicle!(Iss);
    /// define_vehicle!(Soyuz);
    ///
    /// let trans = TranslationalState { position: DVec3::ZERO, velocity: DVec3::ZERO };
    /// let rel = compute_relative_state::<Iss, Iss>(&trans, None, &trans, None);
    /// // Asserting the wrong reference fires the `CompatibleVehicles`
    /// // diagnostic naming `Iss` (found) and `Soyuz` (expected).
    /// let _ = rel.trans.assert_reference::<Soyuz>();
    /// ```
    #[inline]
    pub fn assert_reference<R: Vehicle>(self) -> Self
    where
        (): astrodyn_quantities::diagnostics::CompatibleVehicles<Reference, R>,
    {
        self
    }
}

/// Relative state expressed in the LVLH frame of the reference vehicle.
///
/// The producer ([`compute_lvlh_relative_state`]) always rotates into
/// the reference vehicle's LVLH frame, so the field type
/// `Position<Lvlh<Chief>>` / `Velocity<Lvlh<Chief>>` reflects the
/// frame at compile time. The `<Chief: Vehicle>` parameter names the
/// LVLH chief — mission code that knows the chief at compile time
/// pins it (e.g. `LvlhRelativeState<Iss>`), and the Bevy adapter and
/// runner instantiate `LvlhRelativeState<SelfRef>` because their
/// per-entity storage decides the chief identity at runtime. A
/// consumer holding a `Position<Lvlh<Iss>>` cannot accidentally
/// consume a `Position<Lvlh<Soyuz>>`, and the existing
/// `Lvlh`-vs-`PlanetInertial`/`BodyFrame`/`RootInertial` kind guard
/// remains.
#[derive(Debug, Clone)]
pub struct LvlhRelativeState<Chief: Vehicle> {
    /// Position of subject relative to reference, in the chief
    /// vehicle's LVLH frame (m).
    pub position: Position<Lvlh<Chief>>,
    /// Velocity of subject relative to reference, in the chief
    /// vehicle's LVLH frame (m/s).
    pub velocity: Velocity<Lvlh<Chief>>,
}

impl<Chief: Vehicle> LvlhRelativeState<Chief> {
    /// Type-level witness that this LVLH state carries the caller's
    /// expected chief-vehicle phantom `C`. Compiles only when
    /// `Chief == C`; on mismatch the
    /// [`astrodyn_quantities::diagnostics::CompatibleVehicles`] bound fails
    /// and surfaces a physics-language diagnostic naming the expected
    /// and found chief instead of a `PhantomData<…>` wall.
    ///
    /// The method is a no-op (returns `self`) and has zero runtime
    /// cost; it exists so a consumer holding an
    /// `LvlhRelativeState<Iss>` can refuse — at compile time, with a
    /// physics-language message — a value built for a different chief.
    ///
    /// # Compile-time mismatch
    ///
    /// ```compile_fail
    /// use glam::DVec3;
    /// use astrodyn_quantities::define_vehicle;
    /// use astrodyn::{compute_lvlh_relative_state, LvlhRelativeState};
    ///
    /// define_vehicle!(Iss);
    /// define_vehicle!(Soyuz);
    ///
    /// let r_pos = DVec3::new(6.778e6, 0.0, 0.0);
    /// let r_vel = DVec3::new(0.0, 7.668e3, 0.0);
    /// let s_pos = DVec3::new(6.778e6 + 50.0, 0.0, 0.0);
    /// let s_vel = DVec3::new(0.0, 7.668e3, 0.0);
    /// let lvlh: LvlhRelativeState<Iss> =
    ///     compute_lvlh_relative_state::<Iss>(r_pos, r_vel, s_pos, s_vel);
    /// // Asserting a different chief fires the `CompatibleVehicles`
    /// // diagnostic naming `Iss` (found) and `Soyuz` (expected).
    /// let _ = lvlh.assert_chief::<Soyuz>();
    /// ```
    #[inline]
    pub fn assert_chief<C: Vehicle>(self) -> Self
    where
        (): astrodyn_quantities::diagnostics::CompatibleVehicles<Chief, C>,
    {
        self
    }
}

/// Compute orbital elements from translational state.
///
/// Delegates to [`OrbitalElements::from_cartesian_typed`] via the
/// `astrodyn_quantities::ext` lifters; bit-identical to the deprecated f64
/// path that this wrapper used before Phase 10.
///
/// Returns the planet-erased [`OrbitalElements<SelfPlanet>`] —
/// per-body planet identity in the runner is dynamic (keyed by source
/// index), so the static surface returns a `SelfPlanet`-tagged value.
/// Mission code that knows the planet at compile time should use
/// [`compute_orbital_elements_typed`].
pub fn compute_orbital_elements(
    mu: f64,
    position: DVec3,
    velocity: DVec3,
) -> Result<OrbitalElements<SelfPlanet>, OrbitalError> {
    use astrodyn_quantities::ext::{F64Ext, Vec3Ext};
    use astrodyn_quantities::frame::PlanetInertial;
    // Drive the typed kernel directly with `SelfPlanet`-tagged inputs so
    // the result is already `OrbitalElements<SelfPlanet>` — no relabel
    // step. `OrbitalElements::relabel` is restricted to a `<SelfPlanet>`
    // receiver to prevent silent cross-planet retagging, so going
    // through `<Earth>` and erasing afterwards is no longer available.
    OrbitalElements::<SelfPlanet>::from_cartesian_typed(
        F64Ext::m3_per_s2(mu),
        position.m_at::<PlanetInertial<SelfPlanet>>(),
        velocity.m_per_s_at::<PlanetInertial<SelfPlanet>>(),
    )
}

/// Compute Euler angles from body attitude.
///
/// Converts the left-transformation quaternion to a rotation matrix, then
/// decomposes it into Euler angles using the given sequence. Delegates to
/// [`astrodyn_math::compute_euler_angles_from_matrix_typed`] and unwraps the
/// typed result to radians for f64 storage; bit-identical numerics.
pub fn compute_body_euler_angles(rot: &RotationalState, sequence: EulerSequence) -> [f64; 3] {
    let t_parent_body = rot.quaternion.left_quat_to_transformation();
    let typed = astrodyn_math::compute_euler_angles_from_matrix_typed(&t_parent_body, sequence);
    [
        typed[0].get::<radian>(),
        typed[1].get::<radian>(),
        typed[2].get::<radian>(),
    ]
}

/// Compute the LVLH (Local Vertical Local Horizontal) frame from translational state.
///
/// Delegates to [`astrodyn_math::LvlhFrame::compute`], which owns the single
/// JEOD `LvlhFrame::compute_lvlh_frame()` port. Bit-identical numerics to
/// the typed sibling [`compute_body_lvlh_frame_typed`].
pub fn compute_body_lvlh_frame(position: DVec3, velocity: DVec3) -> LvlhFrame {
    use astrodyn_quantities::ext::Vec3Ext;
    use astrodyn_quantities::frame::{Earth, PlanetInertial};
    // Untyped surface; the typed sibling is generic over the planet, but
    // this f64 entry assumes Earth-centered inertial axes for backward
    // compat with the existing Bevy adapter and tests. Mission code
    // targeting another planet should use the `_typed` sibling with the
    // appropriate `P: Planet` phantom (or the `_for_planet` variant).
    LvlhFrame::compute(
        position.m_at::<PlanetInertial<Earth>>(),
        velocity.m_per_s_at::<PlanetInertial<Earth>>(),
    )
}

pub use astrodyn_math::{compute_body_geodetic, compute_body_solar_beta};

/// Compute the relative state between two bodies.
///
/// Returns the state of `subject` relative to `reference`, with
/// position/velocity expressed in the reference body frame (matching JEOD's
/// `compute_relative_state` convention). When the reference has no rotational
/// state, position/velocity remain in the inertial frame.
///
/// The two phantom parameters name the subject and reference vehicles
/// at compile time — they propagate to [`RelativeState::ang_vel`]
/// (subject body frame) and [`RelativeTranslation::BodyFrame`]
/// (reference body frame) respectively. Mission code that pins both
/// concrete vehicles passes them via turbofish; the Bevy adapter and
/// runner pass `<SelfRef, SelfRef>` because their per-entity storage
/// decides both identities at runtime.
///
/// Derived from JEOD `decr_left` (ref_frame_state.cc):
///   x_{ref:subj} = T_ref * (x_subj - x_ref)
///   v_{ref:subj} = T_ref * (v_subj - v_ref) - ω_ref × x_{ref:subj}
///   w_{ref:subj} = ω_subj - T_{ref→subj} * ω_ref
pub fn compute_relative_state<Subject: Vehicle, Reference: Vehicle>(
    ref_trans: &crate::TranslationalState,
    ref_rot: Option<&RotationalState>,
    subj_trans: &crate::TranslationalState,
    subj_rot: Option<&RotationalState>,
) -> RelativeState<Subject, Reference> {
    let rel_pos_inertial = subj_trans.position - ref_trans.position;
    let rel_vel_inertial = subj_trans.velocity - ref_trans.velocity;

    // Rotate into reference body frame if rotational state available.
    // T_ref transforms from inertial (parent) to reference body frame.
    // The `RelativeTranslation` variant tag tracks which branch fired,
    // so consumers (which previously needed the convention-only struct
    // doc to know the frame) now read the frame at the type level.
    let (trans, t_ref_opt) = if let Some(r_ref) = ref_rot {
        let t_ref = r_ref.quaternion.left_quat_to_transformation();
        let pos = t_ref * rel_pos_inertial;
        // Coriolis correction: v_{ref:subj} = T * Δv - ω_ref × pos
        let vel = t_ref * rel_vel_inertial - r_ref.ang_vel_body.cross(pos);
        (
            RelativeTranslation::BodyFrame {
                position: pos.m_at::<BodyFrame<Reference>>(),
                velocity: vel.m_per_s_at::<BodyFrame<Reference>>(),
            },
            Some(t_ref),
        )
    } else {
        (
            RelativeTranslation::Inertial {
                position: rel_pos_inertial.m_at::<RootInertial>(),
                velocity: rel_vel_inertial.m_per_s_at::<RootInertial>(),
            },
            None,
        )
    };

    // Relative attitude and angular velocity
    let (quaternion, ang_vel) = match (ref_rot, subj_rot) {
        (Some(r_ref), Some(r_subj)) => {
            let q_ref = r_ref.quaternion.to_glam();
            let q_subj = r_subj.quaternion.to_glam();
            let q_rel = q_subj * q_ref.conjugate();

            // ω_rel in subject body frame:
            //   ω_{ref:subj} = ω_subj - T_{ref→subj} * ω_ref
            let t_subj = r_subj.quaternion.left_quat_to_transformation();
            let t_ref = t_ref_opt.unwrap_or_else(|| r_ref.quaternion.left_quat_to_transformation());
            let t_ref_to_subj = t_subj * t_ref.transpose();
            let rel_ang_vel = r_subj.ang_vel_body - t_ref_to_subj * r_ref.ang_vel_body;

            (q_rel, rel_ang_vel)
        }
        _ => (DQuat::IDENTITY, DVec3::ZERO),
    };

    RelativeState {
        trans,
        quaternion,
        // The producer's `rel_ang_vel` is computed in the subject body
        // frame (per the JEOD convention documented above); attach the
        // typed `BodyFrame<Subject>` phantom at the boundary.
        ang_vel: ang_vel.rad_per_s_at::<BodyFrame<Subject>>(),
    }
}

/// Raw kernel: compute relative state expressed in the LVLH frame of the
/// reference vehicle.
///
/// Takes the inertial relative position/velocity and rotates them into the
/// LVLH frame of the reference vehicle. Velocity includes the Coriolis
/// correction for the rotating LVLH frame (ω_LVLH × pos_LVLH), matching
/// JEOD's `compute_relative_state` through the frame tree.
///
/// **Mission code should call [`compute_lvlh_relative_state_typed`] instead.**
/// That typed entry takes `Position<PlanetInertial<P>>`/
/// `Velocity<PlanetInertial<P>>` for both chief and deputy, which the
/// compiler enforces at the call site (preventing chief/deputy frame
/// mixing), and forwards to this kernel via `.raw_si()`. This raw kernel
/// exists for callers that already hold raw `DVec3` from a hot path
/// (no useful phantom available) and explicitly want to skip the typed
/// boundary.
pub fn compute_lvlh_relative_state<Chief: Vehicle>(
    ref_pos: DVec3,
    ref_vel: DVec3,
    subj_pos: DVec3,
    subj_vel: DVec3,
) -> LvlhRelativeState<Chief> {
    let lvlh = compute_body_lvlh_frame(ref_pos, ref_vel);

    // Relative state in inertial frame
    let rel_pos_inertial = subj_pos - ref_pos;
    let rel_vel_inertial = subj_vel - ref_vel;

    // Rotate into LVLH frame using the T_parent_this matrix
    // T_parent_this transforms from parent (inertial) to this (LVLH)
    let pos_lvlh = lvlh.t_parent_this * rel_pos_inertial;
    // Coriolis correction: v_LVLH = T * Δv - ω_LVLH × pos_LVLH
    let vel_lvlh = lvlh.t_parent_this * rel_vel_inertial - lvlh.ang_vel_this.cross(pos_lvlh);

    LvlhRelativeState {
        // The kernel rotates each vector through `lvlh.t_parent_this`,
        // which lands them in the chief vehicle's LVLH frame. The
        // `<Chief>` parameter is named by the caller — mission code
        // that pins a concrete chief gets the cross-vehicle compile
        // guard, while the Bevy adapter and runner pass `SelfRef`
        // because their per-entity storage decides chief identity at
        // runtime.
        position: pos_lvlh.m_at::<Lvlh<Chief>>(),
        velocity: vel_lvlh.m_per_s_at::<Lvlh<Chief>>(),
    }
}

/// Typed sibling of [`compute_lvlh_relative_state`].
///
/// Takes the chief and deputy positions and velocities in the same
/// planet-centered inertial frame `<PlanetInertial<P>>` and returns
/// the LVLH-frame relative state. The structural guarantee is at the
/// call site: passing a body's `Position<IntegrationFrame>` directly,
/// or mixing chief and deputy frames (e.g., chief in `<Earth>` and
/// deputy in `<Moon>`), is a compile error.
///
/// The LVLH frame is anchored to the chief vehicle's planet-centered
/// inertial frame, matching JEOD `LvlhFrame::update` (lvlh_frame.cc),
/// which subscribes to `planet->inertial` via
/// `add_relative_state_to_lvlh_frame`. Bit-identical kernel — calls
/// through to the raw [`compute_lvlh_relative_state`] entry via
/// `.raw_si()` at the boundary.
pub fn compute_lvlh_relative_state_typed<P: astrodyn_quantities::frame::Planet, Chief: Vehicle>(
    ref_pos: Position<astrodyn_quantities::frame::PlanetInertial<P>>,
    ref_vel: Velocity<astrodyn_quantities::frame::PlanetInertial<P>>,
    subj_pos: Position<astrodyn_quantities::frame::PlanetInertial<P>>,
    subj_vel: Velocity<astrodyn_quantities::frame::PlanetInertial<P>>,
) -> LvlhRelativeState<Chief> {
    compute_lvlh_relative_state::<Chief>(
        ref_pos.raw_si(),
        ref_vel.raw_si(),
        subj_pos.raw_si(),
        subj_vel.raw_si(),
    )
}

/// Typed sibling of [`compute_orbital_elements`].
///
/// Delegates to [`OrbitalElements::from_cartesian_typed`] (the typed
/// entry point added by Phase 2). Identical numerics — the typed
/// kernel itself extracts SI base values and calls the same f64
/// implementation. The shared `<P>` phantom on `mu`, `position`,
/// `velocity`, and the returned `OrbitalElements<P>` makes the
/// μ-vs-frame agreement structural.
pub fn compute_orbital_elements_typed<P: astrodyn_quantities::frame::Planet>(
    mu: GravParam<P>,
    position: Position<astrodyn_quantities::frame::PlanetInertial<P>>,
    velocity: Velocity<astrodyn_quantities::frame::PlanetInertial<P>>,
) -> Result<OrbitalElements<P>, OrbitalError> {
    OrbitalElements::<P>::from_cartesian_typed(mu, position, velocity)
}

/// Typed sibling of [`compute_body_euler_angles`].
///
/// Returns Euler angles as a triple of typed [`Angle`]s (radians) using the
/// same JEOD extraction algorithm as the f64 surface. Bit-identical numerics.
pub fn compute_body_euler_angles_typed(
    rot: &RotationalState,
    sequence: EulerSequence,
) -> [Angle; 3] {
    let t_parent_body = rot.quaternion.left_quat_to_transformation();
    astrodyn_math::compute_euler_angles_from_matrix_typed(&t_parent_body, sequence)
}

/// Typed sibling of [`compute_body_lvlh_frame`].
///
/// Accepts inertial-frame typed position/velocity and returns the same
/// [`LvlhFrame`] as the f64 surface. Delegates to
/// [`astrodyn_math::LvlhFrame::compute`], which owns the single JEOD
/// `LvlhFrame::compute_lvlh_frame()` port — keeping one source of truth
/// for the kernel.
pub fn compute_body_lvlh_frame_typed<P: astrodyn_quantities::frame::Planet>(
    position: Position<astrodyn_quantities::frame::PlanetInertial<P>>,
    velocity: Velocity<astrodyn_quantities::frame::PlanetInertial<P>>,
) -> LvlhFrame {
    LvlhFrame::compute(position, velocity)
}

pub use astrodyn_math::{compute_body_geodetic_typed, compute_body_solar_beta_typed};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GeodeticState;
    use astrodyn_quantities::frame::SelfRef;
    use glam::DMat3;

    /// Verify that `compute_body_geodetic` correctly applies the inertial-to-
    /// planet-fixed rotation before computing geodetic coordinates.
    ///
    /// Uses a 90-degree rotation about Z so a position on the inertial +X axis
    /// maps to the planet-fixed +Y axis, producing longitude ≈ π/2.
    /// A transpose/sign error in the rotation would yield longitude ≈ −π/2.
    #[test]
    fn geodetic_with_rotation() {
        use std::f64::consts::FRAC_PI_2;

        const R_EQ: f64 = 6_378_137.0;
        const R_POL: f64 = R_EQ * (1.0 - 1.0 / 298.257_223_563); // JEOD: r_eq * (1 - flat_coeff)

        // 90° rotation about Z: maps +X_inertial → +Y_pfix
        //
        // Row-major rotation matrix for +90° about Z:
        //   [ cos  -sin  0 ]     [ 0  -1  0 ]
        //   [ sin   cos  0 ]  =  [ 1   0  0 ]
        //   [  0     0   1 ]     [ 0   0  1 ]
        //
        // glam DMat3::from_cols takes column-major, so transpose the rows.
        let t_inertial_pfix = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),  // col 0
            DVec3::new(-1.0, 0.0, 0.0), // col 1
            DVec3::new(0.0, 0.0, 1.0),  // col 2
        );

        // ISS-altitude position along inertial +X axis
        let pos_inertial = DVec3::new(R_EQ + 408_000.0, 0.0, 0.0);

        let geo = compute_body_geodetic(pos_inertial, &t_inertial_pfix, R_EQ, R_POL);

        // After rotation, planet-fixed position is along +Y → longitude ≈ π/2
        assert!(
            (geo.longitude - FRAC_PI_2).abs() < 1e-10,
            "expected longitude ≈ π/2, got {}",
            geo.longitude
        );
        // Latitude should be ≈ 0 (equatorial)
        assert!(
            geo.latitude.abs() < 1e-10,
            "expected latitude ≈ 0, got {}",
            geo.latitude
        );
        // Altitude should be ≈ 408 km
        assert!(
            (geo.altitude - 408_000.0).abs() < 1.0,
            "expected altitude ≈ 408000 m, got {}",
            geo.altitude
        );

        // Verify against direct computation to lock down the convention
        let pos_pfix = t_inertial_pfix * pos_inertial;
        let expected = GeodeticState::from_planet_fixed(pos_pfix, R_EQ, R_POL);
        assert_eq!(geo, expected);
    }

    /// `compute_lvlh_relative_state_typed` is a thin boundary wrapper:
    /// it must produce bit-identical output to the raw kernel when fed
    /// the same SI values. The non-trivial chief/deputy geometry below
    /// (offsets in all three axes, eccentric chief orbit) confirms the
    /// boundary preserves numerics while attaching the
    /// `<PlanetInertial<P>>` phantom on every input.
    #[test]
    fn lvlh_relative_typed_round_trip() {
        use astrodyn_quantities::ext::Vec3Ext;
        use astrodyn_quantities::frame::{Earth, PlanetInertial};

        // ISS-style chief, deputy offset by ~100 m radial+along-track.
        let ref_pos = DVec3::new(6.778e6, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, 7.668e3, 0.0);
        let subj_pos = DVec3::new(6.778e6 + 50.0, 80.0, 10.0);
        let subj_vel = DVec3::new(-0.05, 7.668e3 + 0.06, 0.0);

        let raw = compute_lvlh_relative_state::<SelfRef>(ref_pos, ref_vel, subj_pos, subj_vel);
        let typed = compute_lvlh_relative_state_typed::<Earth, SelfRef>(
            ref_pos.m_at::<PlanetInertial<Earth>>(),
            ref_vel.m_per_s_at::<PlanetInertial<Earth>>(),
            subj_pos.m_at::<PlanetInertial<Earth>>(),
            subj_vel.m_per_s_at::<PlanetInertial<Earth>>(),
        );

        // Bit-identical SI values: the typed boundary only attaches a
        // phantom; the math is one shared kernel.
        assert_eq!(typed.position.raw_si(), raw.position.raw_si());
        assert_eq!(typed.velocity.raw_si(), raw.velocity.raw_si());

        // The output `<Chief>` parameter is named by the caller — the
        // raw and typed entries above passed `<SelfRef>` (the wildcard
        // for runtime-resolved adapter chiefs). Mission code that
        // pins a concrete chief vehicle gets the cross-vehicle compile
        // guard at this binding site.
        let _: Position<astrodyn_quantities::frame::Lvlh<astrodyn_quantities::frame::SelfRef>> =
            typed.position;
        let _: Velocity<astrodyn_quantities::frame::Lvlh<astrodyn_quantities::frame::SelfRef>> =
            typed.velocity;
    }

    /// `compute_relative_state` produces the [`RelativeTranslation`]
    /// variant that matches the runtime presence of the reference's
    /// rotational state — `Some(...)` lands in `BodyFrame`, `None`
    /// lands in `Inertial`. The bound type-level annotations on the
    /// destructured `position`/`velocity` are the structural guard:
    /// if the producer ever flipped the variant convention, the
    /// `let _: Position<…> = …` lines would refuse to compile.
    #[test]
    fn relative_state_variant_matches_reference_rotation() {
        use astrodyn_math::JeodQuat;
        use astrodyn_quantities::frame::RootInertial;

        let trans_a = crate::TranslationalState {
            position: DVec3::new(6_778_137.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7668.56, 0.0),
        };
        let trans_b = crate::TranslationalState {
            position: DVec3::new(6_778_237.0, 100.0, -50.0),
            velocity: DVec3::new(0.01, 7668.55, 0.005),
        };
        let rot_a = RotationalState {
            quaternion: {
                let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
                q.normalize();
                q
            },
            ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
        };

        // Some reference rotation → BodyFrame variant, with the typed
        // `Position<BodyFrame<SelfRef>>` recovered by destructure. The
        // `<SelfRef, SelfRef>` turbofish is the canonical
        // runtime-resolved boundary — both per-entity adapters
        // instantiate the producer this way.
        let with_rot =
            compute_relative_state::<SelfRef, SelfRef>(&trans_a, Some(&rot_a), &trans_b, None);
        let RelativeTranslation::BodyFrame { position, velocity } = with_rot.trans else {
            panic!("Some reference rotation must yield BodyFrame variant");
        };
        let _: Position<BodyFrame<SelfRef>> = position;
        let _: astrodyn_quantities::aliases::Velocity<BodyFrame<SelfRef>> = velocity;

        // None reference rotation → Inertial variant, with typed
        // `Position<RootInertial>` recovered by destructure.
        let no_rot = compute_relative_state::<SelfRef, SelfRef>(&trans_a, None, &trans_b, None);
        let RelativeTranslation::Inertial { position, velocity } = no_rot.trans else {
            panic!("None reference rotation must yield Inertial variant");
        };
        let _: Position<RootInertial> = position;
        let _: astrodyn_quantities::aliases::Velocity<RootInertial> = velocity;

        // The branch-agnostic raw escape hatches return the same
        // bit-identical `DVec3` regardless of variant — useful for
        // numerical consumers (e.g. scalar `length()` distance
        // metrics) that don't care about the frame label.
        let no_rot_again =
            compute_relative_state::<SelfRef, SelfRef>(&trans_a, None, &trans_b, None);
        let raw_pos = no_rot_again.trans.position_raw();
        let raw_vel = no_rot_again.trans.velocity_raw();
        assert_eq!(raw_pos, trans_b.position - trans_a.position);
        assert_eq!(raw_vel, trans_b.velocity - trans_a.velocity);
    }

    /// Pinning the producer's `<Subject, Reference>` to two distinct
    /// concrete vehicles propagates each phantom into its expected
    /// slot: `Subject` ends up on `ang_vel`'s body-frame phantom,
    /// `Reference` ends up on the body-frame variant's
    /// position/velocity phantoms. The `let _: …` ascriptions are the
    /// structural guard — a future refactor that swapped the two
    /// phantoms (or dropped one of them) would refuse to typecheck.
    #[test]
    fn relative_state_paired_phantoms_propagate() {
        use astrodyn_math::JeodQuat;
        use astrodyn_quantities::define_vehicle;

        define_vehicle!(Subj);
        define_vehicle!(Ref);

        let trans_a = crate::TranslationalState {
            position: DVec3::new(6_778_137.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7668.56, 0.0),
        };
        let trans_b = crate::TranslationalState {
            position: DVec3::new(6_778_237.0, 100.0, -50.0),
            velocity: DVec3::new(0.01, 7668.55, 0.005),
        };
        let rot_ref = RotationalState {
            quaternion: {
                let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
                q.normalize();
                q
            },
            ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
        };
        let rot_subj = RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::new(0.0, 0.0, 0.001),
        };

        let rel = compute_relative_state::<Subj, Ref>(
            &trans_a,
            Some(&rot_ref),
            &trans_b,
            Some(&rot_subj),
        );

        // Subject phantom propagated into ang_vel.
        let _: astrodyn_quantities::aliases::AngularVelocity<BodyFrame<Subj>> = rel.ang_vel;

        // Reference phantom propagated into the BodyFrame variant.
        let RelativeTranslation::BodyFrame { position, velocity } = rel.trans else {
            panic!("Some reference rotation must yield BodyFrame variant");
        };
        let _: Position<BodyFrame<Ref>> = position;
        let _: astrodyn_quantities::aliases::Velocity<BodyFrame<Ref>> = velocity;
    }

    /// `assert_pair` / `assert_reference` / `assert_chief` are zero-cost
    /// type-witness no-ops that compile when the caller's vehicle
    /// phantoms match the value's. The negative branch is covered by
    /// the per-method `compile_fail` doctests; this `#[test]` confirms
    /// the positive branch round-trips.
    #[test]
    fn vehicle_witness_methods_compile_when_phantoms_match() {
        use astrodyn_math::JeodQuat;
        use astrodyn_quantities::define_vehicle;

        define_vehicle!(Iss);
        define_vehicle!(Soyuz);

        let trans = crate::TranslationalState {
            position: DVec3::new(6_778_137.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7668.56, 0.0),
        };
        let rot = RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        };

        // RelativeState<Iss, Soyuz>::assert_pair::<Iss, Soyuz>()
        // compiles; constructed via the producer with both phantoms
        // pinned to concrete vehicles.
        let rel: RelativeState<Iss, Soyuz> =
            compute_relative_state::<Iss, Soyuz>(&trans, Some(&rot), &trans, Some(&rot));
        let rel = rel.assert_pair::<Iss, Soyuz>();

        // RelativeTranslation<Soyuz>::assert_reference::<Soyuz>() compiles.
        let _ = rel.trans.assert_reference::<Soyuz>();

        // LvlhRelativeState<Iss>::assert_chief::<Iss>() compiles.
        let lvlh: LvlhRelativeState<Iss> = compute_lvlh_relative_state::<Iss>(
            DVec3::new(6.778e6, 0.0, 0.0),
            DVec3::new(0.0, 7.668e3, 0.0),
            DVec3::new(6.778e6 + 50.0, 0.0, 0.0),
            DVec3::new(0.0, 7.668e3, 0.0),
        );
        let _ = lvlh.assert_chief::<Iss>();
    }
}
