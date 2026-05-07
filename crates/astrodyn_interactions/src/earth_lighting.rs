//! Earth lighting model.
//!
//! Computes lighting conditions (eclipse, visibility, albedo) for a vehicle
//! in the vicinity of Earth, considering Sun-Earth and Moon-Earth shadow
//! geometry via angular disk intersections.
//!
//! Eclipse geometry (`circle_intersect`, Sun-Earth/Moon-Earth occlusion) is a
//! faithful port of JEOD `models/environment/earth_lighting/src/earth_lighting.cc`.
//! Earth albedo lighting uses a crude cosine approximation (not a JEOD port).

use astrodyn_quantities::aliases::Position;
use astrodyn_quantities::ext::Vec3Ext;
use astrodyn_quantities::frame::RootInertial;
use glam::DVec3;
use std::f64::consts::PI;

const EPSILON: f64 = 1.0e-12;

/// Apparent parameters of a celestial body as seen from the observer.
///
/// `position` carries `Position<RootInertial>`: the lighting state
/// is always computed against root-inertial source positions
/// (Sun / Moon / Earth at the simulation root), so the field is
/// statically tagged with the root-inertial phantom. A consumer that
/// holds a body's `Position<IntegrationFrame>` cannot subtract it
/// from `LightingBody.position` without going through the typed
/// integration-origin shift first (RF.10).
#[derive(Debug, Clone, Default)]
pub struct LightingBody {
    /// Mean equatorial radius (m).
    pub radius: f64,
    /// Position of the body relative to the observer, in the
    /// simulation's root inertial frame (m).
    pub position: Position<RootInertial>,
    /// Distance from observer to body (m).
    pub distance: f64,
    /// Apparent angular half-angle of body disk (rad).
    pub half_angle: f64,
}

/// Lighting parameters for a body pair (light source occulted by another body).
#[derive(Debug, Clone, Default)]
pub struct LightingParams {
    /// Apparent observation angle from light source to occulting body (rad).
    pub obs_angle: f64,
    /// Phase: fraction of body illuminated (0–1). Set externally; defaults to 1.
    pub phase: f64,
    /// Fraction of body surface occulted (0–1).
    pub occlusion: f64,
    /// Fraction of body surface visible = 1 − occlusion.
    pub visible: f64,
    /// Effective lighting. For eclipse pairs: `phase × visible`.
    /// For `earth_albedo`: set directly from the albedo approximation
    /// (phase/visible are not populated).
    pub lighting: f64,
}

/// Complete earth lighting state computed for one observer position.
#[derive(Debug, Clone, Default)]
pub struct EarthLightingState {
    /// Sun body parameters as seen from observer.
    pub sun_body: LightingBody,
    /// Earth body parameters as seen from observer.
    pub earth_body: LightingBody,
    /// Moon body parameters as seen from observer.
    pub moon_body: LightingBody,
    /// Sun-Earth eclipse: Sun occulted by Earth.
    pub sun_earth: LightingParams,
    /// Moon-Earth eclipse: Moon occulted by Earth.
    pub moon_earth: LightingParams,
    /// Earth albedo lighting.
    pub earth_albedo: LightingParams,
}

/// Compute the area of intersection of two circles.
///
/// Used with angular half-angles (arc-lengths on a unit sphere) to determine
/// what fraction of a celestial disk is eclipsed.
///
/// Returns `true` if the circles intersect (or one contains the other),
/// `false` if they do not overlap.
///
/// Port of JEOD `EarthLighting::circle_intersect`.
pub fn circle_intersect(r_bottom: f64, r_top: f64, d_centers: f64) -> (bool, f64) {
    // Quick check: circles do not intersect
    if d_centers > r_bottom + r_top {
        return (false, 0.0);
    }

    // Check containment: one circle entirely inside the other
    if r_bottom > r_top {
        if d_centers < (r_bottom - r_top) + EPSILON {
            // Top circle completely inside bottom circle
            return (true, PI * r_top * r_top);
        }
    } else if d_centers < (r_top - r_bottom) + EPSILON {
        // Bottom circle completely inside top circle
        return (true, PI * r_bottom * r_bottom);
    }

    // Partial overlap: compute lens-shaped intersection area
    let d_c2 = d_centers * d_centers;
    let r_t2 = r_top * r_top;
    let r_b2 = r_bottom * r_bottom;
    let diff_r2 = r_b2 - r_t2;

    // Law of cosines for intersection angles
    let cos_bottom_ang = ((d_c2 + diff_r2) / (2.0 * d_centers * r_bottom)).clamp(-1.0, 1.0);
    let bottom_ang = cos_bottom_ang.acos();

    let cos_top_ang = ((d_c2 - diff_r2) / (2.0 * d_centers * r_top)).clamp(-1.0, 1.0);
    let top_ang = cos_top_ang.acos();

    // Sum of circular segments
    let top_area = r_t2 * (top_ang - top_ang.sin() * top_ang.cos());
    let bottom_area = r_b2 * (bottom_ang - bottom_ang.sin() * bottom_ang.cos());

    (true, top_area + bottom_area)
}

/// Compute lighting parameters from eclipse geometry.
///
/// Given the intersection of a light source disk and an occulting body disk
/// (as arc-lengths on a unit sphere), computes the occlusion fraction, visible
/// fraction, and effective lighting.
///
/// Port of the inline computation in JEOD `EarthLighting::calc_lighting`
/// (earth_lighting.cc lines 305-308 for sun_earth, 316-320 for moon_earth).
///
/// # Arguments
/// * `source_half_angle` — Apparent half-angle of the light source (rad)
/// * `occluder_half_angle` — Apparent half-angle of the occulting body (rad)
/// * `obs_angle` — Angular separation between source and occluder (rad)
/// * `phase` — Phase factor (0–1); 1.0 for Sun (self-luminous)
pub fn calc_lighting_params(
    source_half_angle: f64,
    occluder_half_angle: f64,
    obs_angle: f64,
    phase: f64,
) -> LightingParams {
    let (_intersects, eclipse_area) =
        circle_intersect(source_half_angle, occluder_half_angle, obs_angle);

    // JEOD: occlusion = eclipse_area / (source_half_angle² * π)
    let source_solid_angle = source_half_angle * source_half_angle * PI;
    let occlusion = if source_solid_angle > 0.0 {
        (eclipse_area / source_solid_angle).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let visible = 1.0 - occlusion;
    // JEOD: lighting = phase * visible
    let lighting = phase * visible;

    LightingParams {
        obs_angle,
        phase,
        occlusion,
        visible,
        lighting,
    }
}

/// Raw kernel: compute earth lighting conditions for an observer.
///
/// Pure function: takes observer position and celestial body positions/radii,
/// returns the complete lighting state. All positions are in the simulation
/// root inertial frame (caller-asserted — the input phantoms are absent).
///
/// **Mission code should call [`compute_earth_lighting_typed`] instead.** That
/// typed entry takes `Position<RootInertial>` for all three positions, which
/// the compiler enforces at the call site; it forwards to this kernel via
/// `.raw_si()`. This raw kernel exists for callers that already hold raw
/// `DVec3` from a hot path (no useful phantom available) and explicitly want
/// to skip the typed boundary.
///
/// Port of JEOD `EarthLighting::calc_lighting`.
///
/// # Arguments
/// * `pos_veh` — Observer (vehicle) position in the simulation root inertial frame (m)
/// * `pos_sun` — Sun position in the simulation root inertial frame (m)
/// * `pos_moon` — Moon position in the simulation root inertial frame (m)
/// * `sun_radius` — Sun equatorial radius (m)
/// * `earth_radius` — Earth equatorial radius (m)
/// * `moon_radius` — Moon equatorial radius (m)
pub fn compute_earth_lighting(
    pos_veh: DVec3,
    pos_sun: DVec3,
    pos_moon: DVec3,
    sun_radius: f64,
    earth_radius: f64,
    moon_radius: f64,
) -> EarthLightingState {
    // Relative positions from observer
    let sun_rel = pos_sun - pos_veh;
    let moon_rel = pos_moon - pos_veh;
    let earth_rel = -pos_veh; // Earth at origin

    let sun_dist = sun_rel.length();
    let moon_dist = moon_rel.length();
    let earth_dist = earth_rel.length();

    // Apparent half-angles (angular radii of disks as seen from observer).
    // Clamp ratio to [0,1] to avoid NaN from asin when observer is inside a body.
    let sun_half = (sun_radius / sun_dist).clamp(0.0, 1.0).asin();
    let moon_half = (moon_radius / moon_dist).clamp(0.0, 1.0).asin();
    let earth_half = (earth_radius / earth_dist).clamp(0.0, 1.0).asin();

    // Sun-Earth observation angle (angle between Sun and Earth as seen from observer)
    let sun_earth_obs = observation_angle(sun_rel, sun_dist, earth_rel, earth_dist);

    // Moon-Earth observation angle
    let moon_earth_obs = observation_angle(moon_rel, moon_dist, earth_rel, earth_dist);

    // Sun eclipsed by Earth (JEOD: calc_lighting lines 303-308)
    let sun_earth = calc_lighting_params(sun_half, earth_half, sun_earth_obs, 1.0);

    // Moon eclipsed by Earth (JEOD: calc_lighting lines 312-320)
    let moon_earth = calc_lighting_params(moon_half, earth_half, moon_earth_obs, 1.0);

    // Earth albedo (JEOD: calc_lighting lines 322-323)
    let earth_albedo_lighting = (sun_earth_obs / PI).abs() * sun_earth.lighting;

    EarthLightingState {
        sun_body: LightingBody {
            radius: sun_radius,
            // Re-tag at the producer boundary: the f64 entry takes raw
            // root-inertial DVec3s for `pos_sun`/`pos_moon`/`pos_veh`,
            // and the relative position therefore lives in the same
            // frame. The typed sibling `compute_earth_lighting_typed`
            // funnels through this same struct via its kernel call.
            position: sun_rel.m_at::<RootInertial>(),
            distance: sun_dist,
            half_angle: sun_half,
        },
        earth_body: LightingBody {
            radius: earth_radius,
            position: earth_rel.m_at::<RootInertial>(),
            distance: earth_dist,
            half_angle: earth_half,
        },
        moon_body: LightingBody {
            radius: moon_radius,
            position: moon_rel.m_at::<RootInertial>(),
            distance: moon_dist,
            half_angle: moon_half,
        },
        sun_earth,
        moon_earth,
        earth_albedo: LightingParams {
            obs_angle: 0.0,
            phase: 0.0,
            occlusion: 0.0,
            visible: 0.0,
            lighting: earth_albedo_lighting,
        },
    }
}

/// Typed sibling of [`compute_earth_lighting`].
///
/// Takes `Position<RootInertial>` for vehicle / sun / moon positions
/// (root-inertial coordinates). Bit-identical kernel — wraps the
/// untyped function via `.raw_si()` at the boundary. The structural
/// guarantee is at the call site: passing a body's
/// `Position<IntegrationFrame>` directly is a compile error, forcing
/// the caller through `IntegOrigin::shift_position` first (RF.10).
#[allow(clippy::too_many_arguments)]
pub fn compute_earth_lighting_typed(
    pos_veh: Position<RootInertial>,
    pos_sun: Position<RootInertial>,
    pos_moon: Position<RootInertial>,
    sun_radius: f64,
    earth_radius: f64,
    moon_radius: f64,
) -> EarthLightingState {
    compute_earth_lighting(
        pos_veh.raw_si(),
        pos_sun.raw_si(),
        pos_moon.raw_si(),
        sun_radius,
        earth_radius,
        moon_radius,
    )
}

/// Compute the observation angle between two directions from the observer.
///
/// Uses `atan2(|cross|, dot)` for numerical stability (avoids acos near 0/π).
fn observation_angle(dir_a: DVec3, dist_a: f64, dir_b: DVec3, dist_b: f64) -> f64 {
    let denom = dist_a * dist_b;
    if denom <= 0.0 {
        return 0.0;
    }
    let cos_obs = dir_a.dot(dir_b) / denom;
    let sin_obs = dir_a.cross(dir_b).length() / denom;
    sin_obs.atan2(cos_obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_no_overlap() {
        let (intersects, area) = circle_intersect(1.0, 1.0, 3.0);
        assert!(!intersects);
        assert_eq!(area, 0.0);
    }

    #[test]
    fn circle_containment_top_inside_bottom() {
        let (intersects, area) = circle_intersect(2.0, 1.0, 0.0);
        assert!(intersects);
        assert!(
            (area - PI).abs() < 1e-12,
            "area should be π·1² = π, got {area}"
        );
    }

    #[test]
    fn circle_containment_bottom_inside_top() {
        let (intersects, area) = circle_intersect(1.0, 2.0, 0.0);
        assert!(intersects);
        assert!(
            (area - PI).abs() < 1e-12,
            "area should be π·1² = π, got {area}"
        );
    }

    #[test]
    fn circle_partial_overlap_symmetric() {
        // Two equal circles overlapping halfway
        let r = 1.0;
        let d = 1.0; // distance = radius
        let (intersects, area) = circle_intersect(r, r, d);
        assert!(intersects);
        // Known result: 2r²(acos(d/2r) - (d/2r)√(1-(d/2r)²))
        let expected = 2.0
            * r
            * r
            * ((d / (2.0 * r)).acos() - (d / (2.0 * r)) * (1.0 - (d / (2.0 * r)).powi(2)).sqrt());
        assert!(
            (area - expected).abs() < 1e-10,
            "area {area} vs expected {expected}"
        );
    }

    #[test]
    fn full_sunlight() {
        // Vehicle far from Earth, Sun on opposite side — no eclipse
        let pos_veh = DVec3::new(1e8, 0.0, 0.0); // 100,000 km from Earth
        let pos_sun = DVec3::new(1.5e11, 0.0, 0.0); // Sun beyond vehicle
        let pos_moon = DVec3::new(0.0, 3.84e8, 0.0); // Moon perpendicular

        let state = compute_earth_lighting(
            pos_veh, pos_sun, pos_moon, 6.96e8,  // Sun radius
            6.371e6, // Earth radius
            1.737e6, // Moon radius
        );

        assert!(
            state.sun_earth.visible > 0.99,
            "Sun should be fully visible, got {}",
            state.sun_earth.visible
        );
        assert!(
            state.sun_earth.occlusion < 0.01,
            "No eclipse expected, got occlusion {}",
            state.sun_earth.occlusion
        );
    }

    #[test]
    fn full_eclipse() {
        // Vehicle directly behind Earth relative to Sun — full shadow
        let pos_sun = DVec3::new(1.5e11, 0.0, 0.0);
        let pos_veh = DVec3::new(-7e6, 0.0, 0.0); // Behind Earth (close)
        let pos_moon = DVec3::new(0.0, 3.84e8, 0.0);

        let state = compute_earth_lighting(
            pos_veh, pos_sun, pos_moon, 6.96e8,  // Sun radius
            6.371e6, // Earth radius
            1.737e6, // Moon radius
        );

        assert!(
            state.sun_earth.occlusion > 0.99,
            "Full eclipse expected, got occlusion {}",
            state.sun_earth.occlusion
        );
        assert!(
            state.sun_earth.visible < 0.01,
            "Sun should be fully occluded, got visible {}",
            state.sun_earth.visible
        );
    }

    #[test]
    fn observation_angle_perpendicular() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let angle = observation_angle(a, 1.0, b, 1.0);
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "90° expected, got {angle}"
        );
    }

    #[test]
    fn observation_angle_same_direction() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let angle = observation_angle(a, 1.0, a, 1.0);
        assert!(angle.abs() < 1e-12, "0° expected, got {angle}");
    }

    /// `LightingBody.position` is `Position<RootInertial>` —
    /// the producer/consumer chain stays in the simulation root
    /// inertial frame. A non-trivial geometry (Sun off-axis, Moon
    /// off-axis, observer not at origin) confirms that the relative
    /// positions surface in the typed field bit-identically to the
    /// raw arithmetic and carry the root-inertial phantom.
    #[test]
    fn lighting_body_position_typed_round_trip() {
        let pos_veh = DVec3::new(1.0e6, 2.0e6, 3.0e5);
        let pos_sun = DVec3::new(1.5e11, -4.0e10, 1.0e9);
        let pos_moon = DVec3::new(0.0, 3.84e8, 1.0e7);
        let state = compute_earth_lighting(pos_veh, pos_sun, pos_moon, 6.96e8, 6.371e6, 1.737e6);

        // Bit-identical SI values: typed wrappers carry the same
        // numbers as the raw subtraction, only the phantom changes.
        assert_eq!(state.sun_body.position.raw_si(), pos_sun - pos_veh);
        assert_eq!(state.moon_body.position.raw_si(), pos_moon - pos_veh);
        assert_eq!(state.earth_body.position.raw_si(), -pos_veh);

        // The typed entry produces the same struct (bit-identical)
        // when fed root-inertial positions, validating the typed
        // boundary doesn't perturb numerics.
        let typed = compute_earth_lighting_typed(
            pos_veh.m_at::<RootInertial>(),
            pos_sun.m_at::<RootInertial>(),
            pos_moon.m_at::<RootInertial>(),
            6.96e8,
            6.371e6,
            1.737e6,
        );
        assert_eq!(
            typed.sun_body.position.raw_si(),
            state.sun_body.position.raw_si()
        );
        assert_eq!(
            typed.moon_body.position.raw_si(),
            state.moon_body.position.raw_si()
        );
        assert_eq!(
            typed.earth_body.position.raw_si(),
            state.earth_body.position.raw_si()
        );
    }
}
