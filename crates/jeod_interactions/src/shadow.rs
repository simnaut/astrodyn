//! Shadow (eclipse) detection.
//!
//! Faithful port of JEOD's `RadiationThirdBody::calculate_shadow()` and
//! `RadiationThirdBody::generate_alpha()` from
//! `models/interactions/radiation_pressure/src/radiation_third_body.cc`.
//!
//! Implements the conical shadow model with:
//! - Region A: vehicle is between source and third body (full sun)
//! - Region B: vehicle is outside the penumbra cone (full sun)
//! - Region C: vehicle is inside the umbra cone (full shadow)
//! - Region D: vehicle is in the antumbra / annular eclipse
//! - Partial eclipse: cubic polynomial approximation of circular disk overlap

use glam::DVec3;

/// Empirically-derived cubic polynomial approximation for the fractional area
/// of a circular disk eclipsed by another circular disk, as a function of the
/// angular size ratio `rho_adj` (smaller/larger) and the eclipse fraction
/// `delta` (0 = first contact, 1 = totality).
///
/// Port of `RadiationThirdBody::generate_alpha()`.
/// Coefficients match JEOD exactly.
fn generate_alpha(rho_adj: f64, delta: f64) -> f64 {
    let rho2 = rho_adj * rho_adj;
    let rho3 = rho2 * rho_adj;

    let a3 = 0.758656 * rho2 + -0.0637441 * rho_adj - 1.08955;
    let a2 = -1.2205 * rho2 + 0.61815 * rho_adj + 1.62909;
    let a1 = 0.43723 * rho2 + -0.52921 * rho_adj + 0.473649;
    let a0 = -0.01242 * rho3 + -0.0036715 * rho2 + 0.0150263 * rho_adj + -0.0078259;

    a3 * delta * delta * delta + a2 * delta * delta + a1 * delta + a0
}

/// Compute the illumination fraction for a vehicle considering shadowing
/// by a celestial body (conical shadow model).
///
/// This is a faithful port of JEOD's `RadiationThirdBody::calculate_shadow()`.
///
/// # Arguments
/// * `vehicle_pos` - Vehicle position in the integration frame (m)
/// * `sun_pos` - Sun (radiation source) position in the integration frame (m)
/// * `body_pos` - Shadow-casting body position (e.g., Earth center) (m)
/// * `body_radius` - Equatorial radius of the shadow-casting body (m)
/// * `source_radius` - Radius of the radiation source (Sun) (m)
///
/// # Returns
/// Illumination fraction: 0.0 = full shadow (umbra), 1.0 = full sunlight.
/// Values between 0 and 1 indicate partial eclipse (penumbra or antumbra).
pub fn compute_shadow_fraction(
    vehicle_pos: DVec3,
    sun_pos: DVec3,
    body_pos: DVec3,
    body_radius: f64,
    source_radius: f64,
) -> f64 {
    // --- Compute the vectors JEOD uses ---
    //
    // JEOD variables (from RadiationThirdBody and RadiationSource):
    //   source_to_third_inrtl = body_pos - sun_pos    (Sun -> third body)
    //   source_to_cg          = vehicle_pos - sun_pos  (Sun -> vehicle)
    //   third_to_cg_inrtl     = source_to_cg - source_to_third_inrtl
    //                         = vehicle_pos - body_pos (third body -> vehicle)

    let source_to_third = body_pos - sun_pos;
    let d_source_to_third = source_to_third.length();

    // JEOD_INV: IN.14 — d_source_to_third > 0 (returns 1.0 if d <= 0)
    if d_source_to_third <= 0.0 {
        // Degenerate: body and source coincide, no shadow possible
        return 1.0;
    }

    let source_to_third_hat = source_to_third / d_source_to_third;

    // Vector from third body to vehicle
    let third_to_cg = vehicle_pos - body_pos;

    // r_par: component of third_to_cg along the source-to-third-body direction
    let r_par = third_to_cg.dot(source_to_third_hat);

    // Region A: If r_par < 0, the vehicle is between the source and the third
    // body (or on the source side). The third body cannot cast a shadow on the
    // vehicle.
    if r_par < 0.0 {
        return 1.0;
    }

    // JEOD_INV: IN.13 — Shadow model: vehicle distance > 0 (returns 0.0 if r_mag2 <= 0)
    // Compute the squared distance between vehicle and third body
    let r_mag2 = third_to_cg.length_squared();
    if r_mag2 <= 0.0 {
        // Vehicle coincides with the third body center
        return 0.0;
    }

    // Perpendicular distance from the vehicle to the source-third-body line
    let r_perp2 = r_mag2 - r_par * r_par;
    let r_perp = if r_perp2 <= 0.0 {
        0.0
    } else {
        r_perp2.sqrt()
    };

    // JEOD_INV: IN.11 — RadiationThirdBody.radius > 0 (degenerate cases handled above)
    // JEOD_INV: IN.12 — RadiationSource.radius > 0 (degenerate cases handled above)
    // --- Precomputed constants (normally set in RadiationThirdBody::initialize) ---
    let r_plus = body_radius + source_radius;
    let r_minus = body_radius - source_radius;
    let r_ratio = body_radius / source_radius;

    // --- Conical shadow geometry (JEOD's calculate_shadow) ---
    //
    // JEOD multiplies through by d_source_to_third to avoid divisions.
    // All boundary tests use:
    //   r_perp_x_d = r_perp * d_source_to_third
    //   radius_x_d = body_radius * d_source_to_third

    let r_perp_x_d = r_perp * d_source_to_third;
    let radius_x_d = body_radius * d_source_to_third;

    // Region B (full sun): vehicle is outside the penumbra cone
    //   r_perp * D >= r_plus * r_par + body_radius * D
    if r_perp_x_d >= r_plus * r_par + radius_x_d {
        return 1.0;
    }

    // Region C (total shadow / umbra): vehicle is inside the umbra cone
    //   r_perp * D <= r_minus * r_par + body_radius * D
    // Note: when body_radius < source_radius, r_minus is negative, so this
    // cone converges. When body_radius > source_radius, it diverges.
    if r_perp_x_d <= r_minus * r_par + radius_x_d {
        return 0.0;
    }

    // Compute angular size ratio (shadow body / source) as seen from vehicle.
    // ang_ratio = (body_radius / source_radius) * (d_source_to_cg / d_third_to_cg)
    //
    // JEOD uses: ang_ratio_2_a = r_ratio * d_source_to_cg
    //            ang_ratio_2   = ang_ratio_2_a^2 / r_mag2
    // where d_source_to_cg = |vehicle_pos - sun_pos|
    let d_source_to_cg = (vehicle_pos - sun_pos).length();
    let ang_ratio_2_a = r_ratio * d_source_to_cg;
    let ang_ratio_2 = ang_ratio_2_a * ang_ratio_2_a / r_mag2;

    // Region D (annular eclipse / antumbra): vehicle is beyond the umbra apex
    // and the third body appears smaller than the source, but is fully
    // contained within the source disk.
    //   r_perp * D <= -(r_minus * r_par + body_radius * D)
    if r_perp_x_d <= -(r_minus * r_par + radius_x_d) {
        return 1.0 - ang_ratio_2;
    }

    // --- Partial eclipse (penumbra) ---
    let ang_ratio = ang_ratio_2.sqrt();

    // delta: fraction of partial eclipse from first contact (0) to totality (1)
    // The numerator is the same for both branches.
    let delta_numer = radius_x_d - r_perp_x_d + (body_radius + source_radius) * r_par;

    if ang_ratio_2 >= 1.0 {
        // Third body has a larger angular size than the source.
        // The source disk is being covered by the larger third body disk.
        let delta = delta_numer / (2.0 * source_radius * r_par);
        1.0 - generate_alpha(1.0 / ang_ratio, delta)
    } else {
        // Source has a larger angular size than the third body.
        // The third body disk transits across the larger source disk.
        let delta = delta_numer / (2.0 * body_radius * (d_source_to_third + r_par));
        1.0 - ang_ratio_2 * generate_alpha(ang_ratio, delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH_RADIUS: f64 = 6_378_137.0;
    const SUN_RADIUS: f64 = 6.98e8;
    const AU: f64 = 1.496e11;

    /// Vehicle directly behind Earth (on the anti-Sun line): full shadow.
    #[test]
    fn vehicle_in_umbra() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO; // Earth at origin
        // Vehicle behind Earth, on the -X side (opposite Sun)
        let vehicle = DVec3::new(-7_000_000.0, 0.0, 0.0); // ~622 km alt, anti-Sun

        let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
        assert_eq!(frac, 0.0, "Vehicle directly behind Earth should be in full shadow");
    }

    /// Vehicle at 90 degrees from Sun-Earth line: full sun.
    #[test]
    fn vehicle_perpendicular_full_sun() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;
        // Vehicle above Earth along +Z
        let vehicle = DVec3::new(0.0, 0.0, 7_000_000.0);

        let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
        assert_eq!(frac, 1.0, "Vehicle perpendicular to Sun-Earth line should be in full sun");
    }

    /// Vehicle on the Sun side of Earth: full sun.
    #[test]
    fn vehicle_sunside_full_sun() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;
        let vehicle = DVec3::new(7_000_000.0, 0.0, 0.0);

        let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
        assert_eq!(frac, 1.0, "Vehicle on Sun side should be in full sun");
    }

    /// Vehicle far behind Earth but offset: full sun (outside penumbra).
    #[test]
    fn vehicle_behind_but_offset() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;
        // Vehicle behind Earth, but far offset in Y
        let vehicle = DVec3::new(-10_000.0, 100_000_000.0, 0.0);

        let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
        assert_eq!(frac, 1.0, "Vehicle far from shadow axis should be in full sun");
    }

    /// Penumbra: vehicle near the edge of the shadow.
    #[test]
    fn vehicle_in_penumbra() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;

        // Vehicle behind Earth at ~1000 km, offset by approximately Earth radius
        // This should put it in the penumbra region
        let vehicle = DVec3::new(-1_000_000.0, EARTH_RADIUS + 1000.0, 0.0);

        let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
        assert!(
            frac > 0.0 && frac < 1.0,
            "Vehicle near shadow edge should be in penumbra, got {frac}"
        );
    }

    /// Shadow fraction is monotonic: moving from shadow axis outward.
    #[test]
    fn shadow_monotonic_with_offset() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;
        let x_behind = -1_000_000.0; // 1000 km behind Earth

        let mut prev_frac = 0.0;
        for y_offset in (0..20).map(|i| (i as f64) * 1_000_000.0) {
            let vehicle = DVec3::new(x_behind, y_offset, 0.0);
            let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
            assert!(
                frac >= prev_frac,
                "Shadow fraction should increase with offset: at y={y_offset}, frac={frac} < prev={prev_frac}"
            );
            prev_frac = frac;
        }
    }

    /// Symmetry: shadow is symmetric about the Sun-body line.
    #[test]
    fn shadow_symmetric() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;
        let x = -500_000.0;

        let frac_y = compute_shadow_fraction(
            DVec3::new(x, 5_000_000.0, 0.0), sun, body, EARTH_RADIUS, SUN_RADIUS,
        );
        let frac_neg_y = compute_shadow_fraction(
            DVec3::new(x, -5_000_000.0, 0.0), sun, body, EARTH_RADIUS, SUN_RADIUS,
        );
        let frac_z = compute_shadow_fraction(
            DVec3::new(x, 0.0, 5_000_000.0), sun, body, EARTH_RADIUS, SUN_RADIUS,
        );

        assert!(
            (frac_y - frac_neg_y).abs() < 1e-12,
            "Shadow should be symmetric in +/-Y"
        );
        assert!(
            (frac_y - frac_z).abs() < 1e-12,
            "Shadow should be symmetric between Y and Z offsets"
        );
    }

    /// generate_alpha returns values in a reasonable range for typical inputs.
    #[test]
    fn generate_alpha_range() {
        // rho_adj=1 (equal size disks), delta=0 (first contact) should be ~0
        let alpha_start = generate_alpha(1.0, 0.0);
        assert!(
            alpha_start.abs() < 0.05,
            "Alpha at first contact should be near 0, got {alpha_start}"
        );

        // rho_adj=1, delta=1 (totality) should be ~1
        let alpha_end = generate_alpha(1.0, 1.0);
        assert!(
            (alpha_end - 1.0).abs() < 0.05,
            "Alpha at totality should be near 1, got {alpha_end}"
        );
    }

    /// Antumbra (Region D): when the vehicle is far behind the umbra apex,
    /// the third body appears smaller than the Sun, producing an annular eclipse.
    #[test]
    fn annular_eclipse_region_d() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;

        // Place vehicle very far behind Earth on the anti-Sun axis.
        // At this distance the umbra cone has converged, and we are in
        // the antumbra. The illumination should be 1 - (angular_ratio)^2.
        // For Earth/Sun at 1 AU, the umbra apex is at roughly
        // R_earth * D_sun / (R_sun - R_earth) ~ 1.38e6 km.
        // Place vehicle at 2e9 m (2 million km) behind Earth.
        let vehicle = DVec3::new(-2.0e9, 0.0, 0.0);

        let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
        // Should be between 0 and 1, and positive (annular eclipse, not total)
        assert!(
            frac > 0.0 && frac < 1.0,
            "Vehicle in antumbra should have partial illumination, got {frac}"
        );
    }

    /// Region A: vehicle between source and third body should be full sun.
    #[test]
    fn vehicle_between_source_and_body() {
        let sun = DVec3::new(AU, 0.0, 0.0);
        let body = DVec3::ZERO;
        // Vehicle between Sun and Earth (positive X, closer to Earth than Sun)
        let vehicle = DVec3::new(AU / 2.0, 0.0, 0.0);

        let frac = compute_shadow_fraction(vehicle, sun, body, EARTH_RADIUS, SUN_RADIUS);
        assert_eq!(frac, 1.0, "Vehicle between source and body should be full sun");
    }
}
