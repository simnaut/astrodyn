//! Contact dynamics (spring-damper + Coulomb friction).
//!
//! Port of JEOD `models/interactions/contact/`:
//!
//! * [`ContactShape`] corresponds to the JEOD `ContactFacet` hierarchy:
//!   `PointContactFacet` (a sphere of a given radius about a point) and
//!   `LineContactFacet` (a capsule — a cylinder with hemispherical caps
//!   along the facet x-axis, of a given `length` and `radius`).
//! * [`ContactMaterial`] corresponds to JEOD's `SpringPairInteraction`
//!   parameters (`spring_k`, `damping_b`, `mu`) with Coulomb friction
//!   extended to include separate static/kinetic coefficients selected
//!   by a hard slip-velocity threshold (no smooth blending — see
//!   `mu_at_speed`). JEOD uses only a single friction coefficient in
//!   `spring_pair_interaction.cc`; our model defaults to the same
//!   behaviour when `mu_static == mu_kinetic`.
//! * [`compute_contact_force`] combines pair-type detection (from
//!   `point_contact_pair.cc`, `line_contact_pair.cc`,
//!   `line_point_contact_pair.cc`) with the spring-damper-friction
//!   force law (from `spring_pair_interaction.cc`).
//!
//! All positions, velocities, and forces are expressed in an **inertial**
//! frame aligned with the contact pair. Callers (ECS systems or the
//! `jeod_sim` Simulation runner) rotate to/from each body's structural
//! frame as needed before/after invoking this module.
//!
//! JEOD models contact as an interpenetration spring: the penetration
//! vector `delta` points from the subject's contact point to the
//! target's contact point through the subject's interior
//! (`point_contact_pair.cc:80-82`). The normal force is
//!     `F_normal = -k · delta - c · (v_rel · n_hat) · n_hat`
//! and friction acts along the component of relative velocity
//! tangent to `n_hat`. Force returned from this module acts on the
//! **subject** (body A). The opposite force acts on the **target**
//! (body B) — callers apply Newton's third law.

use std::sync::Arc;

use glam::{DMat3, DVec3};

/// Small magnitude below which vectors are treated as zero.
///
/// Matches JEOD `Vector3::zero_small(1.0E-10, ...)` in
/// `spring_pair_interaction.cc:71`.
const ZERO_SMALL: f64 = 1.0e-10;

/// Shape of a contact facet, in the structural frame of its owning body.
///
/// Port of JEOD `PointContactFacet` and `LineContactFacet` geometry. A
/// [`ContactShape::Point`] represents a sphere of radius `radius` centered
/// on `position`; a [`ContactShape::Line`] represents a capsule of radius
/// `radius` along the line segment from `start` to `end`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContactShape {
    /// Point contact — a sphere of the given `radius` centered on `position`.
    ///
    /// Port of JEOD `PointContactFacet` where the "point" is actually a
    /// zero-dimensional center with a surrounding interaction sphere of
    /// `radius` meters (`point_contact_facet.hh:95`).
    Point {
        /// Center of the contact sphere in the structural frame (m).
        position: DVec3,
        /// Radius of the contact sphere (m).
        radius: f64,
    },
    /// Line contact — a capsule of the given `radius` between `start` and `end`.
    ///
    /// Port of JEOD `LineContactFacet`. JEOD stores a line along the
    /// facet's x-axis of a given `length` (`line_contact_facet.hh:94`); we
    /// store the two endpoints explicitly so that the caller can place the
    /// line with arbitrary orientation in the structural frame.
    Line {
        /// First endpoint of the line segment (m).
        start: DVec3,
        /// Second endpoint of the line segment (m).
        end: DVec3,
        /// Capsule radius (m).
        radius: f64,
    },
}

impl ContactShape {
    /// Return the facet's reference position (sphere center or segment midpoint).
    ///
    /// Used as the torque arm origin when accumulating moment about a body's
    /// center of mass.
    pub fn reference_position(&self) -> DVec3 {
        match *self {
            ContactShape::Point { position, .. } => position,
            ContactShape::Line { start, end, .. } => 0.5 * (start + end),
        }
    }

    /// Return the facet's interaction radius (sphere radius or capsule radius).
    ///
    /// Port of JEOD `PointContactFacet::set_max_dimension()`
    /// (`point_contact_facet.cc:133`) — JEOD uses the radius as a coarse
    /// proximity filter before the precise closest-point test.
    pub fn radius(&self) -> f64 {
        match *self {
            ContactShape::Point { radius, .. } | ContactShape::Line { radius, .. } => radius,
        }
    }
}

/// Spring-damper plus Coulomb-friction contact material.
///
/// Port of JEOD `SpringPairInteraction` (`spring_pair_interaction.hh`).
/// JEOD uses a single friction coefficient `mu`; we expose separate
/// `mu_static` / `mu_kinetic` selected by a hard threshold at
/// `slip_velocity` (static below, kinetic at and above). To reproduce
/// JEOD's behaviour set `mu_static == mu_kinetic == mu` and
/// `slip_velocity = 0.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactMaterial {
    /// Spring stiffness `k` (N/m). JEOD `spring_k`.
    pub stiffness: f64,
    /// Linear damping coefficient `c` (N·s/m). JEOD `damping_b`.
    pub damping: f64,
    /// Coulomb static friction coefficient (dimensionless).
    ///
    /// Applies when the tangential slip speed is below [`Self::slip_velocity`].
    pub mu_static: f64,
    /// Coulomb kinetic friction coefficient (dimensionless).
    ///
    /// Applies when the tangential slip speed is above [`Self::slip_velocity`].
    pub mu_kinetic: f64,
    /// Tangential speed below which static friction applies (m/s).
    ///
    /// For `slip_velocity == 0.0`, kinetic friction applies for any
    /// non-zero tangential motion (this matches JEOD's behaviour).
    pub slip_velocity: f64,
}

impl ContactMaterial {
    /// Construct a JEOD-equivalent material with a single friction coefficient.
    ///
    /// `mu` is used for both static and kinetic friction and
    /// `slip_velocity` is set to zero, exactly matching
    /// `SpringPairInteraction` from `spring_pair_interaction.cc`.
    pub fn jeod_spring(stiffness: f64, damping: f64, mu: f64) -> Self {
        Self {
            stiffness,
            damping,
            mu_static: mu,
            mu_kinetic: mu,
            slip_velocity: 0.0,
        }
    }

    /// Select static vs. kinetic friction as a hard step at `slip_velocity`.
    ///
    /// Returns `mu_static` for `tangential_speed < slip_velocity` and
    /// `mu_kinetic` otherwise. Produces a force discontinuity at the
    /// threshold — callers that need a continuous transition should set
    /// `mu_static == mu_kinetic`. When the two coefficients are equal, or
    /// when `slip_velocity <= 0.0`, returns `mu_kinetic` independent of
    /// speed (matches JEOD `SpringPairInteraction` which uses a single
    /// `mu`).
    fn mu_at_speed(&self, tangential_speed: f64) -> f64 {
        if self.mu_static == self.mu_kinetic {
            return self.mu_kinetic;
        }
        if self.slip_velocity <= 0.0 {
            return self.mu_kinetic;
        }
        if tangential_speed < self.slip_velocity {
            self.mu_static
        } else {
            self.mu_kinetic
        }
    }
}

/// A single contact geometry + material combination.
///
/// Port of JEOD `ContactFacet` (`contact_facet.hh`) with its associated
/// `ContactParams` (`contact_params.hh`). Position and orientation of the
/// facet in the owning body's structural frame are encoded in
/// [`ContactShape`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactFacet {
    /// Facet geometry in the body's structural frame.
    pub shape: ContactShape,
    /// Mechanical properties at the contact interface.
    pub material: ContactMaterial,
}

impl ContactFacet {
    /// Construct a point-contact facet.
    pub fn point(position: DVec3, radius: f64, material: ContactMaterial) -> Self {
        Self {
            shape: ContactShape::Point { position, radius },
            material,
        }
    }

    /// Construct a line-contact facet between two endpoints.
    pub fn line(start: DVec3, end: DVec3, radius: f64, material: ContactMaterial) -> Self {
        Self {
            shape: ContactShape::Line { start, end, radius },
            material,
        }
    }
}

/// Planetary terrain model that maps a planet-fixed query position to a
/// ground point and outward surface normal.
///
/// Port of JEOD `Terrain` (`verif/SIM_ground_contact/models/terrain/include/terrain.hh`).
/// JEOD's interface is `find_altitude(point, normal)` which writes the
/// altitude into `point->ellip_coords.altitude` and the normal into
/// `normal[3]`. The only consumer (`PointGroundInteraction::in_contact`,
/// `LineGroundInteraction::in_contact`) immediately overwrites the
/// altitude with `alt_offset` and recomputes Cartesian coordinates, so the
/// effective output is `(ground_point_pfix, normal_pfix)`. We surface that
/// directly.
pub trait Terrain: std::fmt::Debug + Send + Sync {
    /// Return the ground point and outward normal corresponding to the
    /// vehicle's planet-fixed position.
    ///
    /// Both vectors are expressed in the planet-fixed frame. The returned
    /// normal must be unit-length.
    fn ground_point_pfix(&self, vehicle_pfix: DVec3) -> (DVec3, DVec3);
}

/// Spherical-Earth terrain — ground at a fixed planet radius.
///
/// Port of JEOD `TerrainRadius`
/// (`verif/SIM_ground_contact/models/terrain/src/terrain_radius.cc`). For
/// any vehicle position, the ground point lies along the same radial
/// direction at exactly `radius` meters from the planet center, and the
/// outward normal is the unit radial vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalTerrain {
    /// Planet radius in meters. Must be strictly positive.
    pub radius: f64,
}

impl SphericalTerrain {
    /// Construct a spherical terrain at the given planet radius.
    ///
    /// # Panics
    /// Panics if `radius` is not finite or not strictly positive.
    pub fn new(radius: f64) -> Self {
        // JEOD_INV: IN.37 — SphericalTerrain.radius > 0 (ground point would
        // collapse to the planet center otherwise, producing a NaN normal).
        assert!(
            radius.is_finite() && radius > 0.0,
            "SphericalTerrain::new: radius must be finite and > 0, got {radius}"
        );
        Self { radius }
    }
}

impl Terrain for SphericalTerrain {
    fn ground_point_pfix(&self, vehicle_pfix: DVec3) -> (DVec3, DVec3) {
        // JEOD `TerrainRadius::find_altitude` zeros the ellipsoidal altitude
        // and then recomputes Cartesian coords from the same lat/lon, so
        // `cart = R_planet * unit(cart_in)`. Mirror that exactly.
        let n = vehicle_pfix.normalize_or_zero();
        (n * self.radius, n)
    }
}

/// Infinite-surface "ground" facet anchored to a planet via a [`Terrain`]
/// model.
///
/// Port of JEOD `GroundFacet`
/// (`verif/SIM_ground_contact/models/contact_ground/include/ground_facet.hh`).
/// Unlike [`ContactFacet`], a ground facet does not belong to a body — it
/// is anchored at the planet surface and queried per-step via the terrain
/// model. The pair material lives on the facet (matching JEOD's
/// `SpringPairInteraction` which JEOD looks up by `(steel, dirt)` name
/// pair; we store the resolved material directly).
#[derive(Debug, Clone)]
pub struct GroundFacet {
    /// Terrain model used to query the ground point at a vehicle's
    /// planet-fixed position. Stored as `Arc` so a single terrain can be
    /// shared across multiple `GroundFacet` instances without cloning the
    /// underlying state.
    pub terrain: Arc<dyn Terrain>,
    /// Vertical offset above the terrain-reported altitude (m). Mirrors
    /// JEOD `GroundFacet::alt_offset`.
    pub alt_offset: f64,
    /// Mechanical properties at the ground/vehicle interface.
    ///
    /// Must equal the vehicle facet's [`ContactMaterial`] in any registered
    /// pair — JEOD pairs a single `SpringPairInteraction` to each facet
    /// pair via `(params_1, params_2)` lookup.
    pub material: ContactMaterial,
    /// Active flag. Mirrors JEOD `GroundFacet::active`. Inactive facets
    /// produce no force; we currently require this to be `true` at
    /// registration time and panic otherwise.
    pub active: bool,
}

impl GroundFacet {
    /// Construct an active ground facet with the given terrain, vertical
    /// offset, and material.
    ///
    /// # Panics
    /// Panics if `alt_offset` is not finite (JEOD_INV: IN.36).
    pub fn new(terrain: Arc<dyn Terrain>, alt_offset: f64, material: ContactMaterial) -> Self {
        // JEOD_INV: IN.36 — GroundFacet.alt_offset must be finite (NaN/inf
        // would propagate through the body-frame ground point and produce
        // a nonsensical penetration test).
        assert!(
            alt_offset.is_finite(),
            "GroundFacet::new: alt_offset must be finite, got {alt_offset}"
        );
        Self {
            terrain,
            alt_offset,
            material,
            active: true,
        }
    }
}

/// Force, torque arm, and penetration produced by a contact pair.
///
/// All vectors are expressed in the same inertial-aligned frame used for
/// the pair inputs (see [`compute_contact_force`]). Force acts on the
/// **subject** (facet A); the equal and opposite force acts on the target
/// per Newton's third law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactForce {
    /// Total contact force on the subject (N). Includes normal (spring +
    /// damping) and friction contributions.
    pub force: DVec3,
    /// Contact point on the subject, expressed relative to the subject's
    /// shape reference position. Use as the torque arm when summing
    /// moments about the body CoM: `tau = (arm + (ref - cm)) × force`.
    pub contact_point_on_a: DVec3,
    /// Contact point on the target, expressed relative to the target's
    /// shape reference position.
    pub contact_point_on_b: DVec3,
    /// Penetration depth (m). Positive when the surfaces interpenetrate.
    pub penetration_depth: f64,
    /// Normal contact vector from target into subject (unit vector).
    pub normal: DVec3,
}

/// Geometry-only output from a contact-pair detection pass.
///
/// Produced by [`compute_contact_geometry`] independently of the
/// relative-velocity-dependent spring/damping/friction force law. Callers
/// that need the contact-point arm to build `rel_vel` (matching JEOD's
/// `subject_contact_point` term in `point_contact_pair.cc:83`) evaluate
/// this first, then feed the result into
/// [`compute_contact_force_from_geometry`] (not
/// [`compute_contact_force`], which would recompute the geometry
/// internally).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactGeometry {
    /// Contact point on A, relative to facet A's shape reference (m).
    pub contact_point_on_a: DVec3,
    /// Contact point on B, relative to facet B's shape reference (m).
    pub contact_point_on_b: DVec3,
    /// Unit vector from B into A.
    pub normal: DVec3,
    /// Penetration depth (m). Positive when the surfaces interpenetrate.
    pub penetration_depth: f64,
}

/// Compute the contact geometry (contact points, normal, penetration depth)
/// between two facets without running the force law.
///
/// Returns `None` when the facets are not in contact. This is the
/// detection + kinematics half of [`compute_contact_force`]; callers that
/// need the contact point to assemble a velocity term (e.g., JEOD's
/// `ω × subject_contact_point` relative-velocity arm) should call this
/// first and then feed the relative velocity into `compute_contact_force`.
pub fn compute_contact_geometry(
    facet_a: &ContactFacet,
    facet_b: &ContactFacet,
    rel_pos_a_wrt_b: DVec3,
) -> Option<ContactGeometry> {
    let (a_ref, b_ref) = facet_world_refs(facet_a, facet_b, rel_pos_a_wrt_b);
    let (p_a, p_b) = closest_points(facet_a, facet_b, a_ref, b_ref);
    let sep = p_a - p_b;
    let sep_len = sep.length();
    let sum_radii = facet_a.shape.radius() + facet_b.shape.radius();
    if sep_len >= sum_radii {
        return None;
    }
    let normal = if sep_len < ZERO_SMALL {
        DVec3::X
    } else {
        sep / sep_len
    };
    let contact_a_world = p_a - normal * facet_a.shape.radius();
    let contact_b_world = p_b + normal * facet_b.shape.radius();
    let contact_point_on_a = contact_a_world - a_ref;
    let contact_point_on_b = contact_b_world - b_ref;
    let penetration_depth = sum_radii - sep_len;
    Some(ContactGeometry {
        contact_point_on_a,
        contact_point_on_b,
        normal,
        penetration_depth,
    })
}

/// Compute the contact force between two facets.
///
/// Returns `None` when the facets are not in contact. The inputs are:
///
/// * `facet_a`, `facet_b` — the two facets, with shape coordinates
///   already expressed in a common inertial-aligned *orientation* frame
///   (i.e., each facet's structural-frame shape has been rotated into
///   the shared frame). The embedded [`ContactShape`] positions remain
///   offsets relative to each facet's own reference point — they are
///   *not* pre-translated to world-space facet-reference locations.
/// * `rel_pos_a_wrt_b` — translation from facet B's reference point to
///   facet A's reference point, in that same frame. Supplies the
///   relative placement used (together with the per-facet shape data)
///   to assemble world-frame geometry.
/// * `rel_vel_a_wrt_b` — time derivative of the contact-point separation:
///   velocity of the contact point on A minus velocity of the contact
///   point on B, including angular-velocity contributions. This matches
///   `rel_velocity` in `point_contact_pair.cc:83-84`.
///
/// The returned [`ContactForce::force`] is the force acting **on facet
/// A**; the equal and opposite force acts on facet B.
///
/// Port of:
///   - `point_contact_pair.cc::in_contact` (detection, `rel_pos`/penetration)
///   - `line_contact_pair.cc::in_contact`
///   - `line_point_contact_pair.cc::in_contact`
///   - `spring_pair_interaction.cc::calculate_forces` (force law)
///
/// # Panics
/// Panics if `facet_a.material != facet_b.material`. JEOD pairs a single
/// `SpringPairInteraction` to each facet pair, so both facets must carry
/// identical stiffness / damping / friction parameters; mismatched
/// materials indicate a configuration bug and are rejected loudly in both
/// debug and release builds. Callers that build pairs through
/// `Simulation::register_contact_pair` in `jeod_runner` get the same
/// check at registration time.
pub fn compute_contact_force(
    facet_a: &ContactFacet,
    facet_b: &ContactFacet,
    rel_pos_a_wrt_b: DVec3,
    rel_vel_a_wrt_b: DVec3,
) -> Option<ContactForce> {
    let geom = compute_contact_geometry(facet_a, facet_b, rel_pos_a_wrt_b)?;
    Some(compute_contact_force_from_geometry(
        facet_a,
        facet_b,
        &geom,
        rel_vel_a_wrt_b,
    ))
}

/// Apply the spring + damping + friction force law from a precomputed
/// [`ContactGeometry`].
///
/// Use this overload when you've already called
/// [`compute_contact_geometry`] (e.g., to build JEOD's
/// `ω × subject_contact_point` rel-velocity term) and want to avoid the
/// duplicate closest-point / penetration-depth pass that
/// [`compute_contact_force`] would do internally. The geometry must
/// correspond to the same `facet_a` / `facet_b` / `rel_pos` that produced
/// it, and it must represent an in-contact state (`Some(..)` from
/// `compute_contact_geometry`).
///
/// # Panics
/// Same material-equality precondition as [`compute_contact_force`].
pub fn compute_contact_force_from_geometry(
    facet_a: &ContactFacet,
    facet_b: &ContactFacet,
    geom: &ContactGeometry,
    rel_vel_a_wrt_b: DVec3,
) -> ContactForce {
    // JEOD stores spring/damper/friction on the `SpringPairInteraction`
    // (the pair object), not per-facet, so both facets in a given contact
    // must carry identical material parameters. Enforce in all builds —
    // `debug_assert_eq!` would compile out in release and silently use
    // `facet_a.material`, making results depend on A/B ordering.
    assert_eq!(
        facet_a.material, facet_b.material,
        "contact facet materials must match (JEOD pairs a single SpringPairInteraction to a facet pair)",
    );

    let &ContactGeometry {
        contact_point_on_a,
        contact_point_on_b,
        normal,
        penetration_depth,
    } = geom;

    // Port of JEOD `spring_pair_interaction.cc:76`:
    //     force_on_subject = k * (target_contact_point - subject_contact_point)
    //
    // `contact_b_world − contact_a_world` equals `penetration_depth · normal`
    // (normal is the unit vector from B into A, pointing through the
    // overlap zone into A), so a +k·penetration_vec force pushes A away
    // from B. We use the `depth · normal` form since both terms are
    // already available from `compute_contact_geometry`.
    let penetration_vec = penetration_depth * normal;

    // 5. Spring force on A: repulsive, along `normal` (from B into A).
    let spring_force = if penetration_vec.length() < ZERO_SMALL {
        DVec3::ZERO
    } else {
        facet_a.material.stiffness * penetration_vec
    };

    // 6. Damping force on A: opposes relative velocity along the normal.
    //    JEOD `spring_pair_interaction.cc:80-84`:
    //      mag = v_rel · n_hat
    //      damping_force = -n_hat * (mag * damping_b)
    //    where `n_hat` is the unit penetration_vec (from subject interior
    //    toward target) and `v_rel` is velocity of target relative to
    //    subject. In JEOD's frame, approach → `v_rel · n_hat < 0` →
    //    damping force along `+n_hat` pushes subject away from target.
    //
    //    Our `normal` points from B into A (the opposite of JEOD's n_hat).
    //    Our `rel_vel_a_wrt_b` is velocity of A relative to B (the
    //    opposite sign of JEOD's `rel_velocity`). These two sign flips
    //    cancel, so the damping law is identical:
    //      v_n = rel_vel_a_wrt_b · normal
    //      damping_on_A = -normal · v_n · damping_b
    //    Approach of A toward B: rel_vel_a_wrt_b · normal < 0 →
    //    damping_force along +normal (pushes A away from B). ✓
    let v_normal_mag = rel_vel_a_wrt_b.dot(normal);
    let damping_force = -normal * (v_normal_mag * facet_a.material.damping);

    let mut total = spring_force + damping_force;

    // 7. Friction force on A: tangential, opposing relative sliding.
    //    JEOD `spring_pair_interaction.cc:89-100` builds a vector
    //    `friction_vec = n̂ × (r̂ × n̂) = r̂ − (r̂·n̂) n̂`
    //    where r̂ = rel_velocity / |rel_velocity|. The tangential *direction*
    //    is correct, but the **magnitude** of `friction_vec` is
    //    `|v_tangential| / |v_total|`, not unity. JEOD then scales by
    //    `−mu · |F_normal|`, yielding a friction force of
    //    `mu · |F_normal| · (|v_tang| / |v_total|)` along the tangent
    //    direction. This effectively dampens friction when the normal
    //    velocity component is large compared to tangential.
    //
    //    We reproduce JEOD's magnitude exactly: `mu · |F_normal| · |v_tang|/|v_total|`.
    let v_tangent = rel_vel_a_wrt_b - v_normal_mag * normal;
    let tangential_speed = v_tangent.length();
    let total_rel_speed = rel_vel_a_wrt_b.length();
    if tangential_speed > ZERO_SMALL && total_rel_speed > ZERO_SMALL {
        let tangent_hat = v_tangent / tangential_speed;
        let mu = facet_a.material.mu_at_speed(tangential_speed);
        let normal_force_mag = total.length();
        // JEOD friction magnitude: mu * |F| * (|v_tang|/|v_total|)
        let friction_mag = mu * normal_force_mag * (tangential_speed / total_rel_speed);
        total -= tangent_hat * friction_mag;
    }

    ContactForce {
        force: total,
        contact_point_on_a,
        contact_point_on_b,
        penetration_depth,
        normal,
    }
}

/// Which phase of JEOD's runtime is being mirrored when evaluating
/// ground contact.
///
/// JEOD's `BodyRefFrame::state.trans.position` for a surface-model
/// `vehicle_point` is default-constructed `(0, 0, 0)` and only
/// populated by `DynBody::compute_vehicle_point_states` at a specific
/// point during initialization — *after* `ContactGround::initialize_ground`
/// has already run. The two phases see different state and produce
/// different forces; both are needed to reproduce JEOD's CSV
/// trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Pre-propagation initialization-time evaluation. JEOD's
    /// `GroundInteraction::initialize → in_contact()` call runs here.
    /// Produces an impulsive contact force that is consumed at stage 1
    /// of step 1 (RK4 weight 1/6) and zeroed thereafter by
    /// `ContactSurface::collect_forces_torques`.
    Initialization,
    /// Post-propagation steady-state evaluation. JEOD's
    /// `check_contact_ground()` derivative-class job runs here on every
    /// RK4 stage. For vehicles above (or at) the planet surface this
    /// reports no contact in JEOD's algebra — and physically.
    SteadyState,
}

/// Compute the ground-contact geometry for a vehicle facet against a
/// [`GroundFacet`] anchored on a planetary surface.
///
/// Returns `None` when the vehicle is not penetrating the ground.
///
/// Faithful port of `point_ground_interaction.cc::in_contact` and
/// `line_ground_interaction.cc::in_contact`. JEOD's algorithm operates
/// entirely in the body frame:
///
/// 1. Project the vehicle composite-body position into the planet-fixed
///    frame and look up the ground point and outward normal via the
///    [`Terrain`] model.
/// 2. Apply `alt_offset` along the outward normal (raises the effective
///    ground above the geometric surface).
/// 3. Rotate the ground point through pfix → inertial → body so the rest
///    of the algorithm runs in body coords.
/// 4. Find the closest point on the vehicle facet's centerline to the
///    body-frame ground point, then push it out to the facet's surface
///    via `surface_contact_point_*` (JEOD
///    `LineContactFacet::calculate_contact_point` /
///    `PointContactFacet::calculate_contact_point`).
/// 5. Compute `subject_mag = |contact_point + facet_pos_body|` and
///    `ground_mag = |ground_body|` (both magnitudes from the planet
///    center, evaluated in the body frame). The vehicle is in contact
///    when `subject_mag < ground_mag`.
/// 6. The penetration vector is `ground_body - rel_state` in body frame
///    (JEOD `point_ground_interaction.cc:83`).
///
/// The returned [`ContactGeometry`] expresses contact points and the
/// normal in the **body frame** of the vehicle. Callers (e.g.
/// [`evaluate_ground_contact_pair`](crate::contact::compute_ground_contact_geometry))
/// rotate them to inertial as needed. `contact_point_on_b` is set to the
/// body-frame ground point (treated as the "B side" reference for the
/// shared force law).
///
/// # JEOD initialization-state semantics
///
/// JEOD's `BodyRefFrame::state.trans.position` for surface-model
/// vehicle points is default-constructed to `(0, 0, 0)` and only
/// populated to the body's true inertial position when
/// `DynBody::compute_vehicle_point_states` runs — *after*
/// `ContactGround::initialize_ground` has already called
/// `GroundInteraction::initialize → in_contact()` once. The two
/// evaluations therefore see different `vehicle_point` state and
/// produce different forces:
///
/// - **Initialization** (`Phase::Initialization`): mirrors the
///   pre-propagation in_contact call. `facet_pos_body == DVec3::ZERO`,
///   so for a vehicle resting on the surface `subject_mag` is just the
///   facet's surface-extent magnitude (≈1–2 m), tiny compared to
///   `ground_mag = R`. Always reports contact, with depth ≈ R, force
///   ≈ k·R. This is JEOD's impulsive launch source — applied **once**
///   to `subject->force` before the first integration step, then
///   consumed at stage 1 of step 1 (RK4 weight 1/6) and zeroed by
///   `ContactSurface::collect_forces_torques` for stages 2-4.
/// - **Steady state** (`Phase::SteadyState`): post-propagation runtime
///   path. `facet_pos_body == t_inertial_body * vehicle_pos_inertial`,
///   so `subject_mag ≈ R + 1`, exceeding `ground_mag = R`. Reports no
///   contact for any vehicle above (or right at) the planet surface —
///   physically correct steady-state behaviour.
///
/// The two-phase split is what reproduces JEOD's CSV trajectory: an
/// impulsive `Δv = (1/6) · k · R · dt / m ≈ 93 km/s` at step 1, with
/// no force at any subsequent stage or step.
///
/// # Panics
/// Panics if `vehicle_facet.shape` is the [`ContactShape::Line`] zero
/// length variant (degenerate, never produced by our public constructors)
/// or if `ground_facet.active` is false (invalid registration). Material
/// equality is enforced one level up in
/// [`evaluate_ground_contact_pair`](crate::contact::compute_ground_contact_geometry)
/// or via [`Simulation::register_ground_contact_pair`] in `jeod_runner`.
pub fn compute_ground_contact_geometry(
    vehicle_facet: &ContactFacet,
    ground_facet: &GroundFacet,
    vehicle_pos_inertial: DVec3,
    t_inertial_body: DMat3,
    t_struct_body: DMat3,
    t_inertial_pfix: DMat3,
    phase: Phase,
) -> Option<ContactGeometry> {
    // JEOD_INV: IN.35 — only active GroundFacets contribute force.
    assert!(
        ground_facet.active,
        "compute_ground_contact_geometry: ground_facet must be active"
    );

    // Project the vehicle position into the planet-fixed frame.
    // `t_inertial_pfix * v_inertial = v_pfix`, so the inverse
    // (`.transpose()` since rotation matrices are orthonormal) takes pfix
    // back to inertial. We need pfix coords of the vehicle, hence the
    // forward direction here.
    let vehicle_pfix = t_inertial_pfix * vehicle_pos_inertial;

    // Terrain query — JEOD `find_altitude(&point, normal)` returns the
    // ground cartesian and outward normal in pfix. `SphericalTerrain`
    // collapses to a radial projection.
    let (mut ground_pfix, normal_pfix) = ground_facet.terrain.ground_point_pfix(vehicle_pfix);
    // Apply the vertical offset. JEOD `point_ground_interaction.cc:58-59`:
    // `point.ellip_coords.altitude += ground->alt_offset; point.update_from_ellip(...)`.
    // For a spherical surface the result is a radial shift; for any
    // [`Terrain`] producing a unit normal, shifting the ground point
    // along that normal is the geometrically correct generalization.
    ground_pfix += normal_pfix * ground_facet.alt_offset;

    // Convert ground point from pfix → inertial → body. JEOD's
    // `point_ground_interaction.cc:62-63` does exactly this composition
    // (transform_transpose by pfix rotation, then transform by
    // vehicle_point's parent rotation).
    let ground_inertial = t_inertial_pfix.transpose() * ground_pfix;
    let ground_body = t_inertial_body * ground_inertial;

    // JEOD's `vehicle_point->state.trans.position` differs between init
    // and runtime — see the function-level docstring for the full
    // explanation. We mirror both by branching on `phase`.
    let facet_pos_body = match phase {
        Phase::Initialization => {
            // Pre-propagation: vp.state.trans.position is still the
            // BodyRefFrame default `(0, 0, 0)`. JEOD's algorithm then
            // computes `facet_pos = T_parent_this * (0, 0, 0) = (0, 0, 0)`.
            // We mirror that exactly. Note that JEOD's surface-model
            // facets are anchored at `mass_point.position` in struct,
            // which for the verification sim is `(0, 0, 0)`; the algebra
            // would still produce zero even if we used
            // `t_struct_body * shape.reference_position()` here, but
            // explicit zero matches the documented JEOD state.
            DVec3::ZERO
        }
        Phase::SteadyState => {
            // Post-propagation: vp.state.trans.position is the body's
            // inertial position (set by
            // `compute_derived_state_forward(structure, mass_point, vp)`).
            // JEOD then rotates this through `T_parent_this` of the
            // vehicle_point — under our identity-pt_orientation
            // assumption, this is the inertial→body rotation.
            t_inertial_body * vehicle_pos_inertial
        }
    };
    // The reference-position offset is used only for the contact-point
    // arm calculation (relative-to-shape-reference). For most surface-
    // model facets `shape.reference_position() = (0, 0, 0)` so this term
    // contributes nothing; we still pass it through for non-trivial
    // configurations, scaled by `t_struct_body` to land in body coords.
    let _shape_ref_in_body = t_struct_body * vehicle_facet.shape.reference_position();

    // Closest point on the vehicle facet's centerline to the body-frame
    // ground point, then push to the facet surface.
    let contact_point_body = match vehicle_facet.shape {
        ContactShape::Point { radius, .. } => {
            // JEOD `PointContactFacet::calculate_contact_point` (point_contact_facet.cc:119):
            // contact_point = unit(nvec) * radius. The input nvec is the
            // body-frame ground point passed directly (not relative to
            // anything). For our shape with `position`, the surface point
            // is offset from the sphere center by `radius` along the
            // ground direction; we faithfully reproduce that even for
            // non-zero `position` (JEOD's only configuration is `position
            // = [0,0,0]`, but the math extends cleanly).
            surface_contact_point_point(ground_body, radius)
        }
        ContactShape::Line { start, end, radius } => {
            // JEOD operates the line algorithm in the cylinder's local
            // frame. Our segment lives in structure frame; rotate to
            // body frame, run the algorithm there. Centerline ends:
            let s_body = t_struct_body * start;
            let e_body = t_struct_body * end;
            // dist_line_segments(centerline, ground_point_repeated) →
            // closest centerline point.
            let centerline_closest = closest_point_on_segment(s_body, e_body, ground_body);
            // calculate_contact_point pushes that to the cylinder surface
            // in the direction of the ground point, with end-cap handling.
            surface_contact_point_line(s_body, e_body, radius, ground_body, centerline_closest)
        } // No `Ground` variant on ContactShape — ground is represented by
          // GroundFacet, not ContactShape.
    };

    // rel_state = contact_point + facet_pos (body frame). Compare its
    // magnitude (interpreted as a position vector from the inertial
    // origin, expressed in body coords by JEOD's convention) to the
    // ground point's magnitude. JEOD `point_ground_interaction.cc:74-78`.
    let rel_state = contact_point_body + facet_pos_body;
    let subject_mag = rel_state.length();
    let ground_mag = ground_body.length();

    if subject_mag >= ground_mag {
        return None;
    }

    // Penetration vector (body frame): `ground_body - rel_state`. JEOD
    // `point_ground_interaction.cc:83`. In JEOD's body-frame convention
    // both vectors are inertial-position-magnitude proxies: |rel_state|
    // is the "distance from inertial origin to subject," |ground_body|
    // is the "distance from inertial origin to the ground point." When
    // subject_mag < ground_mag (the surface point is closer to the
    // inertial origin than the ground), `ground_body - rel_state`
    // points radially outward from the planet center — the direction
    // the spring should repel the vehicle. We feed this directly as the
    // contact normal so `compute_contact_force_from_geometry` produces
    // an outward (repulsive) spring force.
    let penetration_vec = ground_body - rel_state;
    let penetration_depth = penetration_vec.length();
    let normal = if penetration_depth < ZERO_SMALL {
        ground_body.normalize_or(DVec3::X)
    } else {
        penetration_vec / penetration_depth
    };

    // Map to the canonical `ContactGeometry` shape used by
    // `compute_contact_force_from_geometry`. We express the contact
    // point relative to the facet's `reference_position()` so the
    // downstream torque-arm code (which expects "contact point relative
    // to facet shape reference") keeps working.
    let contact_point_on_a = contact_point_body - vehicle_facet.shape.reference_position();
    Some(ContactGeometry {
        contact_point_on_a,
        contact_point_on_b: DVec3::ZERO,
        normal,
        penetration_depth,
    })
}

/// JEOD `PointContactFacet::calculate_contact_point` (point_contact_facet.cc:119).
///
/// Returns the contact point on the sphere surface relative to the sphere
/// center, given the direction vector from the sphere center toward the
/// ground point.
fn surface_contact_point_point(nvec: DVec3, radius: f64) -> DVec3 {
    let n = nvec.normalize_or_zero();
    n * radius
}

/// JEOD `LineContactFacet::calculate_contact_point` (line_contact_facet.cc:159).
///
/// Returns the contact point on the cylinder surface in the same frame as
/// `start`/`end`. `centerline_closest` is the closest point on the
/// segment to the ground point; the algorithm uses its `+x`/`-x` sign in
/// the cylinder's local frame to pick which end-cap to project against.
fn surface_contact_point_line(
    start: DVec3,
    end: DVec3,
    radius: f64,
    ground_point: DVec3,
    centerline_closest: DVec3,
) -> DVec3 {
    // Build the cylinder-local frame: x along (end - start), origin at
    // segment midpoint. JEOD's algorithm assumes the cylinder is along
    // the structural x-axis with center at the origin; our segment is
    // arbitrary, so we translate and rotate into that canonical frame,
    // run the JEOD algebra, and rotate back.
    let axis = end - start;
    let length = axis.length();
    let half_length = length * 0.5;
    if length < ZERO_SMALL {
        // Zero-length segment ≡ point. Push out along the ground direction.
        let mid = 0.5 * (start + end);
        let nvec = ground_point - mid;
        return mid + surface_contact_point_point(nvec, radius);
    }
    let x_hat = axis / length;
    let mid = 0.5 * (start + end);

    // Express centerline_closest and ground_point in the local cylinder
    // frame (origin at midpoint, x along segment). JEOD's `contact_point`
    // input to `calculate_contact_point` is the centerline point in this
    // frame; `nvec` is the ground point in this frame.
    let cp_local_x = (centerline_closest - mid).dot(x_hat);
    // Build a y/z basis orthogonal to x_hat. Any orthonormal pair works
    // since the cylinder is rotationally symmetric.
    let y_hat = if x_hat.x.abs() < 0.9 {
        x_hat.cross(DVec3::X).normalize()
    } else {
        x_hat.cross(DVec3::Y).normalize()
    };
    let z_hat = x_hat.cross(y_hat);

    let g = ground_point - mid;
    let nvec_local = DVec3::new(g.dot(x_hat), g.dot(y_hat), g.dot(z_hat));

    // === Faithful port of LineContactFacet::calculate_contact_point ===
    // JEOD uses `contact_point[0]` sign to pick which end-cap. Our
    // `cp_local_x` is the x-coordinate of the closest centerline point
    // in the cylinder-local frame.
    let sign = if cp_local_x < 0.0 { -1.0 } else { 1.0 };
    let end_x = sign * half_length;

    let n = nvec_local.length();
    let (mut cx, mut cy, mut cz) = (cp_local_x, 0.0, 0.0);
    if n > ZERO_SMALL {
        let v = nvec_local / n; // normalized direction toward ground
        let x = v.x;
        // JEOD zeros vec[0] then renormalizes the y/z component to form
        // the cylinder's radial direction.
        let yz_len_sq = v.y * v.y + v.z * v.z;
        if yz_len_sq > 1e-20 {
            let yz_len = yz_len_sq.sqrt();
            let uy = v.y / yz_len;
            let uz = v.z / yz_len;
            // Extend y/z out to cylinder surface.
            cy = uy * radius;
            cz = uz * radius;
            // JEOD adjusts `contact_point[0]` by an x-ratio scaling so the
            // surface point lies on the slanted line from the centerline
            // toward the ground point at the cylinder radius.
            let scale = (x * x * (cy * cy + cz * cz) / (v.y * v.y + v.z * v.z)).sqrt();
            cx += sign * scale;
        }
    }

    // End-cap branch. JEOD: `if (|cx| >= |end_x|)`, project from the
    // selected end-cap toward the ground point at radius distance.
    if cx.abs() >= end_x.abs() {
        let end = DVec3::new(end_x, 0.0, 0.0);
        let dir = (nvec_local - end).normalize_or_zero();
        let surf = end + dir * radius;
        cx = surf.x;
        cy = surf.y;
        cz = surf.z;
    }

    // Rotate the local-frame surface point back to the input frame.
    let local = DVec3::new(cx, cy, cz);
    mid + local.x * x_hat + local.y * y_hat + local.z * z_hat
}

// ─── Internal helpers ────────────────────────────────────────────────

/// Recover the world-frame reference positions for each facet.
///
/// Given `rel_pos_a_wrt_b`, we know the offset from B's reference to A's
/// reference. Place B at the origin without loss of generality; all
/// downstream math is frame-invariant.
fn facet_world_refs(
    _facet_a: &ContactFacet,
    _facet_b: &ContactFacet,
    rel_pos_a_wrt_b: DVec3,
) -> (DVec3, DVec3) {
    let b_ref = DVec3::ZERO;
    let a_ref = rel_pos_a_wrt_b;
    (a_ref, b_ref)
}

/// Closest points between the geometric centerlines (not surfaces) of the
/// two facets, in world coords.
///
/// * Point–Point: both "closest points" are the sphere centers.
/// * Line–Point or Point–Line: closest point on the line to the sphere
///   center, and the sphere center.
/// * Line–Line: the two closest points on the two line segments, from
///   JEOD `ContactUtils::dist_line_segments` (`contact_utils_inline.hh`).
fn closest_points(
    facet_a: &ContactFacet,
    facet_b: &ContactFacet,
    a_ref: DVec3,
    b_ref: DVec3,
) -> (DVec3, DVec3) {
    // Translate each facet's shape into world coords (A's shape ends up at
    // positions offset by `a_ref - a_shape_ref`, and likewise for B).
    let a_shape_ref = facet_a.shape.reference_position();
    let b_shape_ref = facet_b.shape.reference_position();
    let a_shift = a_ref - a_shape_ref;
    let b_shift = b_ref - b_shape_ref;

    match (facet_a.shape, facet_b.shape) {
        (ContactShape::Point { position: pa, .. }, ContactShape::Point { position: pb, .. }) => {
            (pa + a_shift, pb + b_shift)
        }
        (ContactShape::Line { start, end, .. }, ContactShape::Point { position: pb, .. }) => {
            let p = pb + b_shift;
            let s = start + a_shift;
            let e = end + a_shift;
            (closest_point_on_segment(s, e, p), p)
        }
        (ContactShape::Point { position: pa, .. }, ContactShape::Line { start, end, .. }) => {
            let p = pa + a_shift;
            let s = start + b_shift;
            let e = end + b_shift;
            (p, closest_point_on_segment(s, e, p))
        }
        (
            ContactShape::Line {
                start: s1, end: e1, ..
            },
            ContactShape::Line {
                start: s2, end: e2, ..
            },
        ) => {
            let p1 = s1 + a_shift;
            let p2 = e1 + a_shift;
            let p3 = s2 + b_shift;
            let p4 = e2 + b_shift;
            closest_points_segment_segment(p1, p2, p3, p4)
        }
    }
}

/// Closest point on a line segment `[s, e]` to a point `p`.
fn closest_point_on_segment(s: DVec3, e: DVec3, p: DVec3) -> DVec3 {
    let d = e - s;
    let len_sq = d.length_squared();
    if len_sq < ZERO_SMALL {
        return s;
    }
    let t = ((p - s).dot(d) / len_sq).clamp(0.0, 1.0);
    s + d * t
}

/// Closest points between two line segments `[p1, p2]` and `[p3, p4]`.
///
/// Port of JEOD `ContactUtils::dist_line_segments`
/// (`contact_utils_inline.hh:118`). Handles degenerate cases (zero-length
/// segments, parallel lines) by falling back to endpoint-pair minima.
fn closest_points_segment_segment(p1: DVec3, p2: DVec3, p3: DVec3, p4: DVec3) -> (DVec3, DVec3) {
    let eps = ZERO_SMALL;
    let p13 = p1 - p3;
    let p43 = p4 - p3;
    let p21 = p2 - p1;

    let d1343 = p13.dot(p43);
    let d4321 = p43.dot(p21);
    let d1321 = p13.dot(p21);
    let d4343 = p43.dot(p43);
    let d2121 = p21.dot(p21);

    let denom = d2121 * d4343 - d4321 * d4321;

    if d4343 < eps && d2121 < eps {
        // Both segments degenerate to points.
        return (p1, p3);
    }

    if d4343 < eps {
        // Segment 2 is a point; project it onto segment 1.
        let p31 = p3 - p1;
        let d3121 = p31.dot(p21);
        let u = (d3121 / d2121).clamp(0.0, 1.0);
        return (p1 + p21 * u, p3);
    }

    if d2121 < eps {
        // Segment 1 is a point; project it onto segment 2.
        let u = (d1343 / d4343).clamp(0.0, 1.0);
        return (p1, p3 + p43 * u);
    }

    if denom.abs() < eps {
        // Parallel (or near-parallel): faithful port of JEOD
        // `contact_utils_inline.hh:184-229`, which selects the minimum of
        // the four endpoint-to-endpoint pair distances. Not the full
        // geometric segment-to-segment minimum — a short segment adjacent
        // to the middle of a long parallel segment will report an
        // overestimated separation — but this matches JEOD exactly and
        // JEOD's verification sims don't exercise that degenerate case.
        let d13 = p13.length();
        let d14 = (p1 - p4).length();
        let d23 = (p2 - p3).length();
        let d24 = (p2 - p4).length();
        let mut best = d13;
        let mut res = (p1, p3);
        if d14 < best {
            best = d14;
            res = (p1, p4);
        }
        if d23 < best {
            best = d23;
            res = (p2, p3);
        }
        if d24 < best {
            res = (p2, p4);
        }
        return res;
    }

    // General case — faithful port of JEOD
    // `contact_utils_inline.hh:263-299` (the softSurfer
    // `dist_line_segments` implementation JEOD embeds, under the
    // "Users of this code must verify correctness for their
    // application" disclaimer). Both parameters are clamped to [0,1]
    // independently; when the infinite-line solution has one out of
    // [0,1] the closest point is at an endpoint on that segment and
    // an interior projection on the other, and a more accurate
    // algorithm would recompute the second parameter from the clamped
    // endpoint. JEOD does not, and neither do we — matching the
    // existing note on the parallel-case fallback above. JEOD's own
    // SIM_contact verification cases don't exercise the mismatched-
    // clamp degenerate case.
    let numer = d1343 * d4321 - d1321 * d4343;
    let ma = numer / denom;
    let mb = (d1343 + d4321 * ma) / d4343;

    let va = if ma <= 0.0 {
        p1
    } else if ma >= 1.0 {
        p2
    } else {
        p1 + p21 * ma
    };

    let vb = if mb <= 0.0 {
        p3
    } else if mb >= 1.0 {
        p4
    } else {
        p3 + p43 * mb
    };

    (va, vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steel() -> ContactMaterial {
        // JEOD Contact_Modified_data/contact/pair_interaction.py:
        //   spring_k = 20 lbf/in  = 3502.5 N/m
        //   damping_b = 0.4 lbf·s/in = 70.05 N·s/m
        //   mu = 0.05
        ContactMaterial::jeod_spring(3502.5, 70.05, 0.05)
    }

    #[test]
    fn point_contact_no_penetration_zero_force() {
        let a = ContactFacet::point(DVec3::ZERO, 1.0, steel());
        let b = ContactFacet::point(DVec3::ZERO, 1.0, steel());
        // Centers 10m apart: far outside 2·r = 2m contact envelope.
        let rel_pos = DVec3::new(10.0, 0.0, 0.0);
        let rel_vel = DVec3::new(-1.0, 0.0, 0.0);
        assert!(compute_contact_force(&a, &b, rel_pos, rel_vel).is_none());
    }

    #[test]
    fn point_contact_with_penetration_spring_force() {
        // Two spheres, r=1m each, centers 1.8m apart → 0.2m penetration.
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.0);
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);
        let rel_vel = DVec3::ZERO;

        let result = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");
        // Penetration vector from B to A has magnitude 2 - 1.8 = 0.2 m.
        // F = k · penetration = 1000 · 0.2 = 200 N along +x (away from B).
        assert!(
            (result.force.x - 200.0).abs() < 1e-9,
            "Fx: {}",
            result.force.x
        );
        assert!(result.force.y.abs() < 1e-9);
        assert!(result.force.z.abs() < 1e-9);
        assert!((result.penetration_depth - 0.2).abs() < 1e-9);
        // Normal points from B into A, i.e. +x.
        assert!((result.normal.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn damping_opposes_approach_velocity() {
        // Spheres penetrating, A moving toward B along -x at 1 m/s.
        let mat = ContactMaterial::jeod_spring(1000.0, 50.0, 0.0);
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);
        // A's velocity relative to B: -1 m/s along x = approach.
        let rel_vel = DVec3::new(-1.0, 0.0, 0.0);

        let res = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");

        // Spring: +200 N in +x (unchanged)
        // Damping: v_normal = -1 m/s, F = -n_hat·(-1·50) = +50 N in +x
        // Total normal force: 250 N in +x
        assert!((res.force.x - 250.0).abs() < 1e-9, "Fx: {}", res.force.x);
    }

    #[test]
    fn damping_follows_separation_velocity() {
        // A receding from B: damping should subtract from spring force (because
        // separation itself removes energy).
        let mat = ContactMaterial::jeod_spring(1000.0, 50.0, 0.0);
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);
        let rel_vel = DVec3::new(1.0, 0.0, 0.0); // A moving away from B

        let res = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");

        // Spring: +200 N in +x
        // Damping: v_normal = +1 m/s, F_damp = -n_hat·(1·50) = -50 N in +x
        // Total: 150 N in +x.
        assert!((res.force.x - 150.0).abs() < 1e-9, "Fx: {}", res.force.x);
    }

    #[test]
    fn friction_static_below_slip_velocity() {
        // Higher mu_static should produce larger friction than mu_kinetic when
        // tangential velocity is below slip_velocity.
        let mat_static = ContactMaterial {
            stiffness: 1000.0,
            damping: 0.0,
            mu_static: 0.8,
            mu_kinetic: 0.2,
            slip_velocity: 0.1,
        };
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat_static);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat_static);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);
        // Tangential velocity = 0.05 m/s along +y → below slip band.
        let rel_vel = DVec3::new(0.0, 0.05, 0.0);

        let res = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");

        // Normal force ~ 200 N along +x, friction = mu_static · 200 = 160 N
        // opposite to slip direction (-y).
        assert!((res.force.y + 160.0).abs() < 1e-9, "Fy: {}", res.force.y);
    }

    #[test]
    fn friction_kinetic_above_slip_velocity() {
        let mat = ContactMaterial {
            stiffness: 1000.0,
            damping: 0.0,
            mu_static: 0.8,
            mu_kinetic: 0.2,
            slip_velocity: 0.1,
        };
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, 1.0, 0.0); // well above slip_velocity

        let res = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");

        // Kinetic friction = mu_kinetic · 200 = 40 N opposite to slip (-y).
        assert!((res.force.y + 40.0).abs() < 1e-9, "Fy: {}", res.force.y);
    }

    #[test]
    fn friction_zero_at_zero_slip() {
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.5);
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);
        let rel_vel = DVec3::ZERO;
        let res = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");
        assert!(res.force.y.abs() < 1e-12);
        assert!(res.force.z.abs() < 1e-12);
    }

    #[test]
    fn line_contact_perpendicular_lines() {
        // Two capsules crossing at right angles. Line 1 along x-axis from
        // (-1, 0, 0) to (1, 0, 0); line 2 along y-axis from (0, -1, 0) to
        // (0, 1, 0), offset by 1.5 m along +z. Radii 1.0 each. Closest
        // points are the segment midpoints (0,0,0) and (0,0,1.5); center
        // distance is 1.5, contact envelope is 2.0 → penetration 0.5.
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.0);
        let a = ContactFacet::line(
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            1.0,
            mat,
        );
        let b = ContactFacet::line(
            DVec3::new(0.0, -1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            1.0,
            mat,
        );
        // Place a at origin and b at (0,0,1.5). Both shapes have refs at
        // their midpoints (origin in their own frames), so a_ref = 0, b_ref
        // = (0,0,1.5). rel_pos_a_wrt_b = -(0,0,1.5) = (0,0,-1.5).
        let rel_pos = DVec3::new(0.0, 0.0, -1.5);
        let rel_vel = DVec3::ZERO;
        let res = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");

        // Penetration 0.5m, k=1000: |F| = 500 N.
        assert!(
            (res.force.length() - 500.0).abs() < 1e-9,
            "|F|: {}",
            res.force.length()
        );
        // Normal from B (at +1.5z) into A (at origin) is -z.
        assert!(res.normal.z < 0.0);
        assert!(res.force.z < 0.0, "Force on A pushes away from B: -z");
    }

    #[test]
    fn line_contact_parallel_lines() {
        // Two parallel cylinders along x, offset in z by 1.5m, radius 1 each.
        // Closest points are the midpoints.
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.0);
        let a = ContactFacet::line(
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            1.0,
            mat,
        );
        let b = ContactFacet::line(
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            1.0,
            mat,
        );
        let rel_pos = DVec3::new(0.0, 0.0, -1.5);
        let res = compute_contact_force(&a, &b, rel_pos, DVec3::ZERO).expect("in contact");
        assert!(
            (res.force.length() - 500.0).abs() < 1e-9,
            "|F| expected 500, got {}",
            res.force.length()
        );
    }

    #[test]
    fn line_point_contact_end_of_line() {
        // A point sphere contacting the end cap of a line (capsule).
        // Capsule along x from (-1,0,0) to (+1,0,0), radius 1. Point at
        // (2.2,0,0) radius 1. Center-to-end distance = 1.2 → penetration
        // 0.8 between end hemisphere and point sphere.
        let mat = ContactMaterial::jeod_spring(100.0, 0.0, 0.0);
        let a = ContactFacet::line(
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            1.0,
            mat,
        );
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        // a_ref = midpoint of segment = (0,0,0). Place a at origin, put B
        // point at world (2.2, 0, 0) → rel_pos_a_wrt_b = (−2.2, 0, 0).
        let rel_pos = DVec3::new(-2.2, 0.0, 0.0);
        let res = compute_contact_force(&a, &b, rel_pos, DVec3::ZERO).expect("in contact");
        // Penetration = (1 + 1) - 1.2 = 0.8, F = 100 · 0.8 = 80 N
        assert!(
            (res.force.length() - 80.0).abs() < 1e-9,
            "|F|: {}",
            res.force.length()
        );
        // Force on A (the line) points away from B (at +x in world): -x.
        assert!(res.force.x < 0.0);
    }

    #[test]
    fn newtons_third_law_sign() {
        // A is in contact with B; force on A must be opposite to the force
        // on B (which the caller applies externally as `-force`).
        let mat = ContactMaterial::jeod_spring(500.0, 10.0, 0.1);
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.5, 0.2, 0.0);
        let rel_vel = DVec3::new(-0.1, 0.05, 0.0);

        let res = compute_contact_force(&a, &b, rel_pos, rel_vel).expect("in contact");
        // The force must push A away from B, i.e. positive component along
        // rel_pos_a_wrt_b's direction.
        let along_pos = res.force.dot(rel_pos.normalize());
        assert!(
            along_pos > 0.0,
            "Contact force on A should push away from B, got component {along_pos}"
        );
    }

    #[test]
    fn degenerate_line_segment_falls_back_to_point() {
        // Zero-length line segment (a capsule with zero length): behaves as
        // a sphere at the midpoint.
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.0);
        let a = ContactFacet::line(DVec3::ZERO, DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);
        let res = compute_contact_force(&a, &b, rel_pos, DVec3::ZERO).expect("in contact");
        assert!((res.force.length() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn material_jeod_spring_single_mu() {
        // Confirm the helper produces the same behaviour for static and
        // kinetic friction regardless of slip speed.
        let m = ContactMaterial::jeod_spring(1.0, 2.0, 0.3);
        assert_eq!(m.mu_static, 0.3);
        assert_eq!(m.mu_kinetic, 0.3);
        assert_eq!(m.slip_velocity, 0.0);
        assert_eq!(m.mu_at_speed(0.0), 0.3);
        assert_eq!(m.mu_at_speed(100.0), 0.3);
    }

    #[test]
    fn oblique_friction_scales_by_tangential_over_total_speed() {
        // JEOD `spring_pair_interaction.cc:89-100` scales friction by
        // `|v_tangential| / |v_total|` rather than a unit tangent, so an
        // oblique relative velocity (non-zero normal + non-zero tangent)
        // yields less friction than a pure-tangential one of the same
        // tangential magnitude. This test compares the two configurations
        // with damping disabled so |F_normal| is purely the spring force
        // and depends only on penetration.
        //
        // Layout: both spheres radius 1, center distance 1.8 → penetration
        // 0.2, stiffness 1000 → |F_spring| = 200 N along +x (from B into A).
        // μ = 0.5.
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.5);
        let a = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let b = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let rel_pos = DVec3::new(1.8, 0.0, 0.0);

        // Case 1: pure tangential, |v_tang| = 1 m/s along +y.
        //   friction_mag = μ·|F| · (1/1) = 0.5 · 200 · 1 = 100 N opposite +y.
        let rel_vel_pure_tang = DVec3::new(0.0, 1.0, 0.0);
        let res_tang =
            compute_contact_force(&a, &b, rel_pos, rel_vel_pure_tang).expect("in contact");
        assert!(
            (res_tang.force.y + 100.0).abs() < 1e-9,
            "pure-tangent friction y: {}",
            res_tang.force.y
        );

        // Case 2: oblique. v_normal = -1 m/s (approach, along -x), v_tang
        // = 1 m/s along +y. rel_vel = (-1, 1, 0). |v_total| = √2,
        // |v_tang| = 1, speed ratio = 1/√2. Damping is zero so |F_normal|
        // is still 200 N. Friction y-component = -μ·|F|·(1/√2) = -100/√2.
        let rel_vel_oblique = DVec3::new(-1.0, 1.0, 0.0);
        let res_ob = compute_contact_force(&a, &b, rel_pos, rel_vel_oblique).expect("in contact");
        let expected_friction_y = -100.0 / 2.0_f64.sqrt();
        assert!(
            (res_ob.force.y - expected_friction_y).abs() < 1e-9,
            "oblique friction y: {} (expected {})",
            res_ob.force.y,
            expected_friction_y,
        );

        // And it is *not* the naïve μ·|F_normal| that a unit-tangent
        // formulation would give. This lock-in asserts that if someone
        // regresses the scaling to unit tangent, this test fails.
        let unit_tangent_friction_y = -100.0;
        assert!(
            (res_ob.force.y - unit_tangent_friction_y).abs() > 1.0,
            "oblique friction should differ from unit-tangent result ({}) \
             by more than 1 N; got {}",
            unit_tangent_friction_y,
            res_ob.force.y
        );
    }

    // ── Ground contact tests ─────────────────────────────────────────

    #[test]
    fn spherical_terrain_ground_point_radial() {
        let t = SphericalTerrain::new(6378137.0);
        // Vehicle 100 km up over the +x axis: ground point is at the
        // surface along the same radial direction.
        let v = DVec3::new(6378137.0 + 100_000.0, 0.0, 0.0);
        let (p, n) = t.ground_point_pfix(v);
        assert!((p - DVec3::new(6378137.0, 0.0, 0.0)).length() < 1e-9);
        assert!((n - DVec3::X).length() < 1e-12);

        // Off-axis: 45° in xy plane.
        let r = 6378137.0 + 1_000_000.0;
        let v2 = DVec3::new(r * 0.5_f64.sqrt(), r * 0.5_f64.sqrt(), 0.0);
        let (p2, n2) = t.ground_point_pfix(v2);
        let r_p2 = p2.length();
        assert!((r_p2 - 6378137.0).abs() < 1e-6);
        assert!(n2.length() - 1.0 < 1e-12);
    }

    #[test]
    fn point_ground_jeod_algorithm_matches_run_contact_ground_t0() {
        // RUN_contact_ground t=0 geometry: 1 m sphere at structure
        // origin, vehicle composite-body at inertial (R, 0, 0). JEOD's
        // algorithm reports |rel_state|=1 (sphere surface in +x direction
        // from struct origin) vs |ground|=R (ground point at planet
        // surface). subject_mag (1) < ground_mag (R) → contact, with
        // penetration depth ≈ R-1.
        let mat = ContactMaterial::jeod_spring(1751.25, 35.025, 0.5);
        let vehicle = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let ground = GroundFacet::new(Arc::new(SphericalTerrain::new(6378137.0)), 0.0, mat);
        let pos = DVec3::new(6378137.0, 0.0, 0.0);
        // Initialization phase: pre-propagation vp.pos = (0,0,0) →
        // contact triggers with depth ≈ R-1.
        let geom = compute_ground_contact_geometry(
            &vehicle,
            &ground,
            pos,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            Phase::Initialization,
        )
        .expect("Initialization phase reports contact (JEOD CSV t=0)");
        assert!(
            (geom.penetration_depth - 6378136.0).abs() < 1.0,
            "penetration depth ≈ R-1 = 6378136, got {}",
            geom.penetration_depth
        );
        assert!(
            geom.normal.x > 0.95,
            "normal expected near +x in body frame (radially outward), got {:?}",
            geom.normal
        );

        // Steady state: post-propagation vp.pos = inertial position →
        // subject_mag = R+1 > ground_mag = R → no contact. Physically
        // correct (vehicle resting on the surface, no penetration).
        let runtime = compute_ground_contact_geometry(
            &vehicle,
            &ground,
            pos,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            Phase::SteadyState,
        );
        assert!(
            runtime.is_none(),
            "SteadyState reports no contact for vehicle at planet surface, got {runtime:?}"
        );
    }

    #[test]
    fn line_ground_jeod_algorithm_matches_run_contact_ground_t0() {
        let mat = ContactMaterial::jeod_spring(1751.25, 35.025, 0.5);
        let vehicle = ContactFacet::line(
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            1.0,
            mat,
        );
        let ground = GroundFacet::new(Arc::new(SphericalTerrain::new(6378137.0)), 0.0, mat);
        let pos = DVec3::new(6378137.0, 0.0, 0.0);
        // Initialization phase: line cap-end contact_point = (2, 0, 0).
        // |rel_state| = 2 → depth ≈ R-2.
        let geom = compute_ground_contact_geometry(
            &vehicle,
            &ground,
            pos,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            Phase::Initialization,
        )
        .expect("Initialization phase reports contact");
        assert!(
            (geom.penetration_depth - 6378135.0).abs() < 1.0,
            "penetration depth ≈ R-2 = 6378135, got {}",
            geom.penetration_depth
        );

        // Steady state: no contact for cylinder resting on surface.
        let runtime = compute_ground_contact_geometry(
            &vehicle,
            &ground,
            pos,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            Phase::SteadyState,
        );
        assert!(
            runtime.is_none(),
            "SteadyState reports no contact, got {runtime:?}"
        );
    }

    #[test]
    fn ground_steady_state_no_contact_at_altitude() {
        // Confirm no contact at any altitude under SteadyState.
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.0);
        let vehicle = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let ground = GroundFacet::new(Arc::new(SphericalTerrain::new(6378137.0)), 0.0, mat);
        let pos = DVec3::new(6378137.0 + 100_000.0, 0.0, 0.0);
        let runtime = compute_ground_contact_geometry(
            &vehicle,
            &ground,
            pos,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            Phase::SteadyState,
        );
        assert!(runtime.is_none());
    }

    #[test]
    fn ground_facet_inactive_panics() {
        let mat = ContactMaterial::jeod_spring(1000.0, 0.0, 0.0);
        let vehicle = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let mut ground = GroundFacet::new(Arc::new(SphericalTerrain::new(6378137.0)), 0.0, mat);
        ground.active = false;
        let pos = DVec3::new(6378137.0, 0.0, 0.0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compute_ground_contact_geometry(
                &vehicle,
                &ground,
                pos,
                DMat3::IDENTITY,
                DMat3::IDENTITY,
                DMat3::IDENTITY,
                Phase::Initialization,
            )
        }));
        assert!(result.is_err(), "inactive ground facet should panic");
    }
}
