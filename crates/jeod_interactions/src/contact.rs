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
//!   extended to include separate static/kinetic coefficients and a
//!   slip-velocity transition band. JEOD uses only a single friction
//!   coefficient in `spring_pair_interaction.cc`; our model defaults
//!   to the same behaviour when `mu_static == mu_kinetic`.
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

use glam::DVec3;

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
    /// Applies when the tangential slip speed is below [`slip_velocity`].
    pub mu_static: f64,
    /// Coulomb kinetic friction coefficient (dimensionless).
    ///
    /// Applies when the tangential slip speed is above [`slip_velocity`].
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

/// Compute the contact force between two facets.
///
/// Returns `None` when the facets are not in contact. The inputs are:
///
/// * `facet_a`, `facet_b` — the two facets, with positions/orientations
///   already expressed in a common inertial-aligned frame (not the body
///   structural frame). That is, [`ContactShape`] positions are the
///   facet reference points in world space, not in the body's
///   structural frame.
/// * `rel_pos_a_wrt_b` — the offset of facet A relative to facet B's
///   reference. Used to assemble each facet's world-frame geometry from
///   the per-facet shape data.
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
pub fn compute_contact_force(
    facet_a: &ContactFacet,
    facet_b: &ContactFacet,
    rel_pos_a_wrt_b: DVec3,
    rel_vel_a_wrt_b: DVec3,
) -> Option<ContactForce> {
    // JEOD stores spring/damper/friction on the `SpringPairInteraction`
    // (the pair object), not per-facet, so both facets in a given contact
    // must carry identical material parameters. Enforce in all builds —
    // `debug_assert_eq!` would compile out in release and silently use
    // `facet_a.material`, making results depend on A/B ordering.
    assert_eq!(
        facet_a.material, facet_b.material,
        "contact facet materials must match (JEOD pairs a single SpringPairInteraction to a facet pair)",
    );

    // 1. Resolve contact geometry into world-frame endpoints/centers.
    //    `a_ref`/`b_ref` are the facet reference positions in world coords.
    //    For a Line facet, the endpoints are the shape's start/end offset
    //    by the same rigid translation as the reference position.
    let (a_ref, b_ref) = facet_world_refs(facet_a, facet_b, rel_pos_a_wrt_b);

    // 2. Closest points between the two shapes.
    let (p_a, p_b) = closest_points(facet_a, facet_b, a_ref, b_ref);

    // 3. Penetration check: surfaces overlap when center distance is less
    //    than the sum of the effective radii at the closest points.
    //    For a Point or Line facet the effective radius is `radius()`.
    let sep = p_a - p_b;
    let sep_len = sep.length();
    let sum_radii = facet_a.shape.radius() + facet_b.shape.radius();
    if sep_len >= sum_radii {
        return None;
    }

    // 4. Contact normal: unit vector from B into A, consistent with JEOD's
    //    convention that `penetration_vector = target_contact_point -
    //    subject_contact_point` points from subject into target. We use
    //    the opposite convention internally (normal from B to A) so that
    //    positive penetration produces a force pushing A away from B.
    let normal = if sep_len < ZERO_SMALL {
        // Degenerate centers overlap; fall back to an arbitrary axis.
        // JEOD does not have a defined behaviour here; we pick +x to keep
        // the result deterministic.
        DVec3::X
    } else {
        sep / sep_len
    };

    // Contact points on each shape's surface along `normal`.
    // `contact_point_on_a` is the closest point on A's surface to B, in
    // world coords relative to a_ref: walk from the closest point p_a
    // toward B by `radius_a` along -normal.
    let contact_a_world = p_a - normal * facet_a.shape.radius();
    let contact_b_world = p_b + normal * facet_b.shape.radius();
    let contact_point_on_a = contact_a_world - a_ref;
    let contact_point_on_b = contact_b_world - b_ref;

    // Penetration depth: how far the two surfaces have overlapped.
    //
    // Port of JEOD `spring_pair_interaction.cc:76`:
    //     force_on_subject = k * (target_contact_point - subject_contact_point)
    //
    // JEOD's `penetration_vector` is the vector from the subject's
    // surface point to the target's surface point. When the surfaces
    // overlap, it points from outside-A toward B's surface (i.e.,
    // from A's closest-to-B surface point into the overlap region,
    // toward B). Scaling by `+k` then pushes the subject away from the
    // target.
    //
    // We compute the force on **A** (taking A = JEOD's subject). The
    // equivalent of JEOD's `penetration_vector` is:
    //     penetration_vec = contact_b_world - contact_a_world
    // which points from A's surface outward toward B's surface through
    // the overlap zone. For the sphere-sphere case with A at +x of B,
    // A's surface faces -x (toward B) and B's surface faces +x (toward A);
    // with A at (1.8, 0, 0), radius 1, and B at origin, radius 1:
    //     contact_a_world = (0.8, 0, 0)   ← A's leftmost point
    //     contact_b_world = (1.0, 0, 0)   ← B's rightmost point
    //     penetration_vec = (0.2, 0, 0)   ← from A's surface toward +x
    // `+k · penetration_vec` yields a force on A along +x, pushing A
    // away from B. ✓
    let penetration_vec = contact_b_world - contact_a_world;
    let penetration_depth = sum_radii - sep_len;

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

    Some(ContactForce {
        force: total,
        contact_point_on_a,
        contact_point_on_b,
        penetration_depth,
        normal,
    })
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

    // General case.
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
}
