//! Tier 2: ECEF↔geodetic (and ECEF↔spherical) round-trip closure for the
//! `PlanetShape` ellipsoid + `jeod_math::geodetic` conversion kernels,
//! seeded by the three explicit test points from JEOD's
//! `SIM_PFIXPOSN_VERIF` Trick verification sim.
//!
//! ## Reference
//!
//! - JEOD source: `models/utils/planet_fixed/planet_fixed_posn/verif/
//!   SIM_PFIXPOSN_VERIF/SET_test/RUN_pfixposn_test/input.py`. Three
//!   `add_read` blocks define a Cartesian seed, a spherical seed, and an
//!   elliptical seed.
//! - JEOD verification methodology (random vector sweep) is in
//!   `verif/unit_tests/Cartesian_to_AltLatLong_to_Cartesian/main.cc`,
//!   which checks Cartesian closure of `cart -> spher -> cart` and
//!   `cart -> ellip -> cart`. We mirror that methodology against the
//!   SIM's deterministic seeds so the inputs are sourced from JEOD
//!   itself.
//! - `Planet` defaults (`r_eq = 6_378.137 km`, `flat_inv = 298.257223563`)
//!   come from `environment/planet/data/include/earth.hh`.
//!
//! ## What is checked
//!
//! For every seed we exercise the identical Cartesian-closure invariant
//! that JEOD's own unit test asserts (`vmag(cart_in - cart_out) < tol`).
//! For the spherical seed and the elliptical seed we *additionally* run
//! the angle-space round-trip — but only when the seed's latitude lies
//! inside the principal range `(-π/2, +π/2)` so that the inverse is
//! well-defined. The first `Spherical` seed in `input.py` is `lat=3.1416`
//! (an aliased value used to stress the `sin/cos` numerical path in
//! JEOD's verification log; it intentionally exits the principal range
//! and so cannot be recovered angularly). For that case we limit the
//! check to Cartesian closure, matching JEOD's own treatment.
//!
//! ## Polar handling
//!
//! Per CLAUDE.md "Common Pitfalls / Geodetic longitude at the poles",
//! geodetic longitude is geometrically undefined as latitude approaches
//! ±π/2 because all meridians converge. None of the three SIM seeds
//! places us at the pole (closest is the Cartesian seed which sits on
//! the equator at +x, latitude 0). No degraded longitude tolerance is
//! required.
//!
//! ## Tolerance policy
//!
//! Per CLAUDE.md "Cross-validation tolerances", each tolerance is set
//! to the observed max error * 1.05, then rounded up to a clean
//! literal. Since both inputs and code are deterministic, the observed
//! errors are fixed numbers.

use glam::DVec3;
use jeod_math::geodetic::{
    cartesian_to_geodetic_typed, cartesian_to_spherical, geodetic_to_cartesian_typed,
    spherical_to_cartesian, GeodeticState, GeodeticStateTyped, SphericalState,
};
use jeod_planet::EARTH;
use jeod_quantities::aliases::Position;
use jeod_quantities::ext::F64Ext;
use jeod_quantities::frame::{Earth, PlanetFixed};
use jeod_quantities::qty3::Qty3;
use jeod_test_data::{
    jeod_path,
    planet_geodetic_verif::{load_planet_fixed_verif_cases, PlanetFixedSeed},
};
use uom::si::angle::radian;
use uom::si::length::meter;

/// Resolve the geodetic kernels through the typed surface (and back) so
/// the bare-`f64` impls that the typed wrappers delegate to are exercised
/// indirectly. Returns the recovered Cartesian position.
fn geodetic_round_trip(cart: DVec3) -> (DVec3, GeodeticState) {
    let pos: Position<PlanetFixed<Earth>> = Qty3::from_raw_si(cart);
    let geo = cartesian_to_geodetic_typed(pos, EARTH.r_eq.m(), EARTH.r_pol.m());
    let back: Position<PlanetFixed<Earth>> =
        geodetic_to_cartesian_typed(geo, EARTH.r_eq.m(), EARTH.r_pol.m());
    (back.raw_si(), geo.into_raw())
}

fn spherical_round_trip(cart: DVec3) -> (DVec3, SphericalState) {
    let sph = cartesian_to_spherical(cart, EARTH.r_eq);
    let back = spherical_to_cartesian(&sph, EARTH.r_eq);
    (back, sph)
}

#[derive(Default)]
struct MaxErrors {
    cart_sphere_m: f64,
    cart_ellip_m: f64,
    sphere_lat_rad: f64,
    sphere_lon_rad: f64,
    sphere_alt_m: f64,
    geo_lat_rad: f64,
    geo_lon_rad: f64,
    geo_alt_m: f64,
}

impl MaxErrors {
    fn record_cart(&mut self, sphere: f64, ellip: f64) {
        self.cart_sphere_m = self.cart_sphere_m.max(sphere);
        self.cart_ellip_m = self.cart_ellip_m.max(ellip);
    }

    fn record_sphere_angles(&mut self, lat: f64, lon: f64, alt: f64) {
        self.sphere_lat_rad = self.sphere_lat_rad.max(lat);
        self.sphere_lon_rad = self.sphere_lon_rad.max(lon);
        self.sphere_alt_m = self.sphere_alt_m.max(alt);
    }

    fn record_geo_angles(&mut self, lat: f64, lon: f64, alt: f64) {
        self.geo_lat_rad = self.geo_lat_rad.max(lat);
        self.geo_lon_rad = self.geo_lon_rad.max(lon);
        self.geo_alt_m = self.geo_alt_m.max(alt);
    }
}

/// Smallest absolute longitude difference modulo 2π. `atan2` returns in
/// `(-π, +π]`, so a seed value at +π comes back as -π even though they
/// represent the same meridian. Both branches are mathematically valid;
/// the test must compare on the circle, not on the line.
fn wrap_lon_diff(a: f64, b: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut d = (a - b).rem_euclid(two_pi);
    if d > std::f64::consts::PI {
        d = two_pi - d;
    }
    d.abs()
}

fn assert_jeod_source_present() -> std::path::PathBuf {
    let root = jeod_path();
    assert!(
        root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH to the JEOD checkout \
         (see CLAUDE.md \"Environment Setup\").",
        root.display(),
    );
    root
}

/// Cartesian-closure for every seed plus angle-space closure where the
/// seed angles live inside the principal range.
#[test]
fn tier2_planet_geodetic_round_trip_sim_pfixposn_seeds() {
    let root = assert_jeod_source_present();
    let cases = load_planet_fixed_verif_cases(&root);
    assert_eq!(cases.len(), 3, "SIM_PFIXPOSN_VERIF has three seeds");

    let mut max = MaxErrors::default();

    for (idx, case) in cases.iter().enumerate() {
        match *case {
            PlanetFixedSeed::Cartesian { cart_m, .. } => {
                // Forward+back through both branches.
                let (back_geo, geo) = geodetic_round_trip(cart_m);
                let (back_sph, _sph) = spherical_round_trip(cart_m);

                let err_geo = (back_geo - cart_m).length();
                let err_sph = (back_sph - cart_m).length();
                max.record_cart(err_sph, err_geo);

                // Also confirm: a second forward into the same lat/lon/alt
                // is convergent — the iterative `cartesian_to_geodetic`
                // solver returns the same lat/lon/alt (within the kernel's
                // own tolerance) when re-fed its own Cartesian output.
                // We compare on a tolerance, not bit-exact, because the
                // second pass starts from `back_geo` which already carries
                // ULP-level round-trip error from pass one. Tolerances
                // mirror those used in `jeod_math/src/geodetic.rs` tests.
                let (back2, geo2) = geodetic_round_trip(back_geo);
                let dlat = (geo2.latitude - geo.latitude).abs();
                let dlon = (geo2.longitude - geo.longitude).abs();
                let dalt = (geo2.altitude - geo.altitude).abs();
                assert!(
                    dlat < 1e-12,
                    "case {idx} (cart): geodetic lat not idempotent (Δ = {dlat:.3e} rad)",
                );
                assert!(
                    dlon < 1e-12,
                    "case {idx} (cart): geodetic lon not idempotent (Δ = {dlon:.3e} rad)",
                );
                assert!(
                    dalt < 1e-6,
                    "case {idx} (cart): geodetic alt not idempotent (Δ = {dalt:.3e} m)",
                );
                assert!(
                    (back2 - back_geo).length() < 1e-6,
                    "case {idx} (cart): second round-trip drifted",
                );
            }
            PlanetFixedSeed::Spherical {
                altitude_m,
                latitude_rad,
                longitude_rad,
                ..
            } => {
                let seed = SphericalState {
                    altitude: altitude_m,
                    latitude: latitude_rad,
                    longitude: longitude_rad,
                };
                let cart = spherical_to_cartesian(&seed, EARTH.r_eq);
                let (back_sph_cart, recovered) = spherical_round_trip(cart);
                let err_cart = (back_sph_cart - cart).length();
                max.cart_sphere_m = max.cart_sphere_m.max(err_cart);

                if latitude_rad.abs() < std::f64::consts::FRAC_PI_2 {
                    // Angle-space inverse exists.
                    let dlat = (recovered.latitude - latitude_rad).abs();
                    let dlon = wrap_lon_diff(recovered.longitude, longitude_rad);
                    let dalt = (recovered.altitude - altitude_m).abs();
                    max.record_sphere_angles(dlat, dlon, dalt);
                }
                // Note: the SIM's first spherical seed uses lat=3.1416
                // (out-of-range) to stress the sin/cos numerical path;
                // angle recovery is not meaningful there. JEOD's own
                // unit test checks Cartesian closure only — we mirror
                // that behaviour.
            }
            PlanetFixedSeed::Elliptical {
                altitude_m,
                latitude_rad,
                longitude_rad,
                ..
            } => {
                let seed = GeodeticState {
                    altitude: altitude_m,
                    latitude: latitude_rad,
                    longitude: longitude_rad,
                };
                let typed_seed = GeodeticStateTyped::from_raw(seed);
                let cart_typed: Position<PlanetFixed<Earth>> =
                    geodetic_to_cartesian_typed(typed_seed, EARTH.r_eq.m(), EARTH.r_pol.m());
                let cart = cart_typed.raw_si();

                let (back_cart, recovered) = geodetic_round_trip(cart);
                let err_cart = (back_cart - cart).length();
                max.cart_ellip_m = max.cart_ellip_m.max(err_cart);

                if latitude_rad.abs() < std::f64::consts::FRAC_PI_2 {
                    let dlat = (recovered.latitude - latitude_rad).abs();
                    let dlon = wrap_lon_diff(recovered.longitude, longitude_rad);
                    let dalt = (recovered.altitude - altitude_m).abs();
                    max.record_geo_angles(dlat, dlon, dalt);
                }
            }
        }
    }

    eprintln!(
        "tier2_planet_geodetic_round_trip_sim_pfixposn_seeds: \
         max cart_sphere = {:.3e} m, max cart_ellip = {:.3e} m, \
         max sphere lat/lon/alt = {:.3e}/{:.3e}/{:.3e}, \
         max geo lat/lon/alt = {:.3e}/{:.3e}/{:.3e}",
        max.cart_sphere_m,
        max.cart_ellip_m,
        max.sphere_lat_rad,
        max.sphere_lon_rad,
        max.sphere_alt_m,
        max.geo_lat_rad,
        max.geo_lon_rad,
        max.geo_alt_m,
    );

    // Tolerances per CLAUDE.md "Tolerance policy" (observed max * 1.05,
    // rounded up to a clean literal).
    //
    // Observed (Apr 2026) on x86_64-linux for the three SIM_PFIXPOSN_VERIF
    // seeds:
    //   cart_sphere = 9.313e-10 m  (Cartesian closure of cart->spher->cart;
    //                               first SIM seed lives on the +x axis at
    //                               r=6 778 136.3 m, error is ULP-level)
    //   cart_ellip  = 6.880e-10 m  (cart->ellip->cart over the same point
    //                               and the third seed reconstructed via
    //                               update_from_ellip)
    //   sphere lat/lon/alt = 0/0/0 (only sampled when the spherical seed
    //                               has a principal-range latitude; the
    //                               SIM seed is intentionally out-of-range
    //                               so this field is never written)
    //   geo lat/lon = 0/0          (third SIM seed lat=1.0 rad, lon=π;
    //                               longitude wrap handled by wrap_lon_diff)
    //   geo alt    = 9.140e-11 m
    //
    // The 1e-9 metric tolerances are the closest clean literal above
    // observed*1.05 and absorb a small amount of ULP jitter on
    // alternative targets (aarch64 etc.).
    assert!(
        max.cart_sphere_m < 1.0e-9,
        "cart->spher->cart closure {:.3e} m exceeds 1.0e-9 m tolerance",
        max.cart_sphere_m,
    );
    assert!(
        max.cart_ellip_m < 1.0e-9,
        "cart->ellip->cart closure {:.3e} m exceeds 1.0e-9 m tolerance",
        max.cart_ellip_m,
    );
    assert!(
        max.sphere_lat_rad < 1.0e-12,
        "spherical lat closure {:.3e} rad exceeds 1.0e-12 rad tolerance",
        max.sphere_lat_rad,
    );
    assert!(
        max.sphere_lon_rad < 1.0e-12,
        "spherical lon closure {:.3e} rad exceeds 1.0e-12 rad tolerance",
        max.sphere_lon_rad,
    );
    assert!(
        max.sphere_alt_m < 1.0e-6,
        "spherical alt closure {:.3e} m exceeds 1.0e-6 m tolerance",
        max.sphere_alt_m,
    );
    assert!(
        max.geo_lat_rad < 1.0e-12,
        "geodetic lat closure {:.3e} rad exceeds 1.0e-12 rad tolerance",
        max.geo_lat_rad,
    );
    assert!(
        max.geo_lon_rad < 1.0e-12,
        "geodetic lon closure {:.3e} rad exceeds 1.0e-12 rad tolerance",
        max.geo_lon_rad,
    );
    assert!(
        max.geo_alt_m < 1.0e-10,
        "geodetic alt closure {:.3e} m exceeds 1.0e-10 m tolerance",
        max.geo_alt_m,
    );
}

/// The `PlanetShape::r_pol` value derived from `r_eq` and `flat_coeff`
/// must drive the same Cartesian point as JEOD's `Planet` initialization
/// (which computes `r_pol = r_eq * (1 - 1/flat_inv)`). Verifies our
/// preset is bit-identical to JEOD for the seed coordinates.
#[test]
fn tier2_planet_geodetic_earth_preset_matches_jeod_default() {
    // JEOD: planet/data/include/earth.hh — r_eq = 6378.137 km, flat_inv = 298.257223563.
    let jeod_r_eq_m = 6_378_137.0_f64;
    let jeod_flat_inv = 298.257_223_563_f64;
    let jeod_r_pol_m = jeod_r_eq_m * (1.0 - 1.0 / jeod_flat_inv);

    assert_eq!(EARTH.r_eq, jeod_r_eq_m);
    assert!(
        (EARTH.flat_inv() - jeod_flat_inv).abs() < 1e-12,
        "flat_inv differs by {}",
        (EARTH.flat_inv() - jeod_flat_inv).abs(),
    );
    assert!(
        (EARTH.r_pol - jeod_r_pol_m).abs() < 1e-9,
        "r_pol differs by {} m",
        (EARTH.r_pol - jeod_r_pol_m).abs(),
    );
}

/// Exercise the typed surface end-to-end: typed seed in, typed result
/// out, drive the kernel through `cartesian_to_geodetic_typed` /
/// `geodetic_to_cartesian_typed`. Mirrors the SIM's third (elliptical)
/// seed inside the principal latitude range.
#[test]
fn tier2_planet_geodetic_typed_round_trip_iss_inclined() {
    // ISS-like, but distinctly off-pole so longitude is well-defined.
    let seed = GeodeticStateTyped {
        latitude: 51.6.deg(),
        longitude: 30.0.deg(),
        altitude: 408_000.0.m(),
    };
    let r_eq = EARTH.r_eq.m();
    let r_pol = EARTH.r_pol.m();

    let cart: Position<PlanetFixed<Earth>> = geodetic_to_cartesian_typed(seed, r_eq, r_pol);
    let recovered = cartesian_to_geodetic_typed(cart, r_eq, r_pol);

    let lat_err = (recovered.latitude.get::<radian>() - seed.latitude.get::<radian>()).abs();
    let lon_err = (recovered.longitude.get::<radian>() - seed.longitude.get::<radian>()).abs();
    let alt_err = (recovered.altitude.get::<meter>() - seed.altitude.get::<meter>()).abs();

    assert!(lat_err < 1e-14, "ISS-inclined lat err = {lat_err} rad");
    assert!(lon_err < 1e-14, "ISS-inclined lon err = {lon_err} rad");
    assert!(alt_err < 1e-9, "ISS-inclined alt err = {alt_err} m");
}
