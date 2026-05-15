// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Neutral-atmosphere density, temperature, pressure, and wind models.
//!
//! Pure-Rust port of JEOD's atmosphere models from
//! `models/environment/atmosphere/`. All evaluations are framed by the
//! [`AtmosphereState`] return type — density (kg/m^3), temperature (K),
//! pressure (Pa), and inertial-frame wind velocity (m/s). The crate has zero
//! Bevy dependency; orchestration lives in `astrodyn`.
//!
//! ## Models
//!
//! - [`exponential`] — `rho = rho_0 * exp(-(h - h_0) / H)`. A simple,
//!   single-parameter scale-height model intended as a fallback and as a
//!   sanity-check baseline for tests. Not physically accurate above ~100 km
//!   where diffusive equilibrium takes over.
//! - [`met`] — Marshall Engineering Thermosphere model. Faithful port of
//!   JEOD's MET implementation, which is built on the Jacchia 1970/1971
//!   thermosphere papers (SAO Special Reports No. 313 and 332). MET combines
//!   temperature-profile integration via Gauss quadrature with seasonal-
//!   latitude density corrections and is the production model for LEO drag.
//!
//! ## Co-rotation wind
//!
//! [`compute_corotation_wind`] (and its typed sibling
//! [`compute_corotation_wind_typed`]) implement JEOD's
//! `WindVelocity::update_wind()` for a planet whose angular velocity points
//! along the inertial Z axis: `wind = omega × r`. Aerodynamic drag computes
//! its relative velocity against this co-rotating wind, not against the bare
//! inertial frame.
//!
//! ## Typed quantities
//!
//! [`AtmosphereState<P>`] is parameterized over the atmosphere planet `P`
//! so the wind vector carries `Velocity<PlanetInertial<P>>` directly.
//! `AtmosphereState<Earth>` and `AtmosphereState<Mars>` are distinct types
//! and the compiler refuses to feed Mars's wind to an Earth drag kernel.
//! Producers that determine the planet at runtime (`evaluate_atmosphere`,
//! the runner's per-body atmospheric_state) return
//! `AtmosphereState<SelfPlanet>`; mission code that knows the planet at
//! compile time uses the typed sibling `evaluate_atmosphere_typed::<P>()`
//! and consumes `AtmosphereState<P>`. The boundary between the two is the
//! [`AtmosphereState::<SelfPlanet>::relabel`] method, restricted to the
//! planet-erased variant so a planet-pinned `AtmosphereState<Earth>`
//! cannot accidentally be retagged as `AtmosphereState<Mars>`. JEOD source
//! paths used: `models/environment/atmosphere/MET/` and the broader
//! `models/environment/atmosphere/` model directory; the wind-frame
//! convention is documented in
//! `models/environment/atmosphere/base_atmos/include/wind_velocity.hh`
//! ("the inertial frame of the planet causing the wind velocity").
//!
//! ## Example
//!
//! Default exponential atmosphere with Earth sea-level reference values:
//!
//! ```
//! use astrodyn_atmosphere::exponential::ExponentialAtmosphere;
//!
//! let atmos = ExponentialAtmosphere::default();
//! let sea_level = atmos.density(0.0);
//! assert!((sea_level.density - 1.225).abs() < 1e-9);
//!
//! // Density falls off exponentially with altitude.
//! let one_scale_height = atmos.density(8_500.0);
//! assert!(one_scale_height.density < sea_level.density);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use astrodyn_quantities::prelude::*;

pub mod exponential;
pub mod met;

use core::marker::PhantomData;
use glam::DVec3;
use uom::si::f64::{MassDensity, Pressure, ThermodynamicTemperature};
use uom::si::mass_density::kilogram_per_cubic_meter;
use uom::si::pressure::pascal;
use uom::si::thermodynamic_temperature::kelvin;

/// Atmospheric state at a given position.
///
/// Output of an atmosphere model evaluation. All quantities are in SI units.
/// The phantom `P: Planet` ties the wind vector to the planet whose
/// inertial frame the corotation `ω × r` is computed in. Density,
/// temperature, and pressure are scalar quantities and frame-agnostic;
/// the planet phantom only structurally guards the `wind` field. See the
/// crate-level docs for the producer/consumer split between
/// `AtmosphereState<SelfPlanet>` (registry/runner storage) and
/// `AtmosphereState<P>` (planet-pinned mission code).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereState<P: Planet> {
    /// Atmospheric density in kg/m^3.
    pub density: f64,
    /// Temperature in K.
    pub temperature: f64,
    /// Pressure in N/m^2 (Pa).
    pub pressure: f64,
    /// Wind velocity in the planet `P`'s inertial frame (m/s).
    ///
    /// JEOD's `WindVelocity::update_wind` writes the corotation term
    /// `ω × r` in the planet's inertial frame; the typed wrapper makes
    /// that frame explicit at the field type. A drag consumer that
    /// subtracts this from the vehicle velocity must hold a
    /// `Velocity<PlanetInertial<P>>` on the same `P` — the compiler
    /// refuses to mix Earth's wind with a Mars-frame velocity.
    pub wind: Velocity<PlanetInertial<P>>,
    _p: PhantomData<P>,
}

impl<P: Planet> Default for AtmosphereState<P> {
    fn default() -> Self {
        Self {
            density: 0.0,
            temperature: 0.0,
            pressure: 0.0,
            wind: Velocity::<PlanetInertial<P>>::from_raw_si(DVec3::ZERO),
            _p: PhantomData,
        }
    }
}

impl<P: Planet> AtmosphereState<P> {
    /// Construct an `AtmosphereState<P>` from raw SI scalars and an
    /// already-typed wind vector. The wind's `PlanetInertial<P>`
    /// phantom must agree with the struct's `P`; the compiler enforces
    /// this directly.
    #[inline]
    pub const fn new(
        density: f64,
        temperature: f64,
        pressure: f64,
        wind: Velocity<PlanetInertial<P>>,
    ) -> Self {
        Self {
            density,
            temperature,
            pressure,
            wind,
            _p: PhantomData,
        }
    }

    /// Construct an `AtmosphereState<P>` from raw scalar fields plus a
    /// raw `DVec3` wind, attaching the `PlanetInertial<P>` phantom to
    /// the wind at the boundary. **The caller asserts** that `wind` is
    /// expressed in `PlanetInertial<P>` (the corotation-wind frame).
    /// Used at producer sites that work in raw `DVec3` internally
    /// (exponential, MET) and re-wrap on exit.
    #[inline]
    pub fn from_raw(density: f64, temperature: f64, pressure: f64, wind_raw: DVec3) -> Self {
        Self {
            density,
            temperature,
            pressure,
            wind: Velocity::<PlanetInertial<P>>::from_raw_si(wind_raw),
            _p: PhantomData,
        }
    }

    /// Typed accessor: atmospheric density as `uom::si::f64::MassDensity` (kg/m^3).
    #[inline]
    pub fn density_typed(&self) -> MassDensity {
        MassDensity::new::<kilogram_per_cubic_meter>(self.density)
    }

    /// Typed accessor: atmospheric temperature as
    /// `uom::si::f64::ThermodynamicTemperature` (kelvin).
    #[inline]
    pub fn temperature_typed(&self) -> ThermodynamicTemperature {
        ThermodynamicTemperature::new::<kelvin>(self.temperature)
    }

    /// Typed accessor: atmospheric pressure as `uom::si::f64::Pressure` (pascal).
    #[inline]
    pub fn pressure_typed(&self) -> Pressure {
        Pressure::new::<pascal>(self.pressure)
    }
}

impl AtmosphereState<SelfPlanet> {
    /// Relabel a planet-erased ([`SelfPlanet`]) atmospheric state as
    /// belonging to a specific planet `Q`.
    ///
    /// Restricted to `impl AtmosphereState<SelfPlanet>` so it can only
    /// retag a state that is already planet-erased — a planet-pinned
    /// `AtmosphereState<Earth>` cannot accidentally be relabeled as
    /// `AtmosphereState<Mars>` via this method. **Boundary-only escape
    /// hatch** for relabel sites where the planet identity is determined
    /// at runtime (e.g. wrapping a raw `evaluate_atmosphere` result, the
    /// runner's `body.atmospheric_state` storage, the Bevy
    /// `AtmosphericStateC` Component which is parameterized by
    /// `SelfPlanet`). Mission code that knows the atmosphere planet at
    /// compile time should reach for `evaluate_atmosphere_typed::<Q>`
    /// directly, which produces `AtmosphereState<Q>` without going
    /// through this relabel.
    ///
    /// A genuine `<P>` → `<Q>` retag for two distinct named planets is
    /// almost never the right operation (atmospheric state is always
    /// computed against a specific planet); if you need it for a
    /// different reason, add a separate, clearly-named escape hatch
    /// instead of widening this impl block.
    #[inline]
    pub fn relabel<Q: Planet>(self) -> AtmosphereState<Q> {
        AtmosphereState::<Q> {
            density: self.density,
            temperature: self.temperature,
            pressure: self.pressure,
            wind: Velocity::<PlanetInertial<Q>>::from_raw_si(self.wind.raw_si()),
            _p: PhantomData,
        }
    }
}

/// Compute atmospheric co-rotation wind velocity in the inertial frame.
///
/// Port of JEOD `WindVelocity::update_wind()` with uniform omega scale.
/// Wind is the cross product of the planet's angular velocity vector (Z-axis)
/// with the vehicle's inertial position: `wind = omega × r`.
///
/// For Earth's Z-axis rotation this simplifies to:
///   wind = [-omega * y, omega * x, 0]
///
/// # Arguments
/// * `omega` - Planet angular velocity in rad/s (Earth: 7.292115146706388e-5)
/// * `inertial_pos` - Vehicle position in the inertial frame (m)
// JEOD_INV: AT.04 — wind velocity computed as omega × position (co-rotation)
pub fn compute_corotation_wind(omega: f64, inertial_pos: DVec3) -> DVec3 {
    DVec3::new(-omega * inertial_pos.y, omega * inertial_pos.x, 0.0)
}

/// Typed variant of [`compute_corotation_wind`].
///
/// Generic over the atmosphere planet `P`: wind is the planet's
/// corotation velocity in the planet's own inertial frame, so both
/// position and velocity carry the `PlanetInertial<P>` phantom rather
/// than `RootInertial`. Bit-identical kernel — wraps the raw
/// computation via `.raw_si()` at the boundary.
#[inline]
pub fn compute_corotation_wind_typed<P: Planet>(
    omega: uom::si::f64::AngularVelocity,
    pos: Position<PlanetInertial<P>>,
) -> Velocity<PlanetInertial<P>> {
    use uom::si::angular_velocity::radian_per_second;
    let w = omega.get::<radian_per_second>();
    let p = pos.raw_si();
    compute_corotation_wind(w, p).m_per_s_at::<PlanetInertial<P>>()
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary"
)]
mod tests {
    use super::*;

    #[test]
    fn corotation_wind_at_equator() {
        let omega = 7.292115146706388e-5;
        let r = 6_778_137.0; // ~400 km altitude
        let pos = DVec3::new(r, 0.0, 0.0); // on equator, X-axis

        let wind = compute_corotation_wind(omega, pos);

        // omega × [r, 0, 0] = [0, omega*r, 0]
        assert!(wind.x.abs() < 1e-10);
        assert!((wind.y - omega * r).abs() < 1e-6);
        assert!(wind.z.abs() < 1e-10);

        // ~494 m/s at 400 km
        assert!(wind.length() > 400.0 && wind.length() < 600.0);
    }

    #[test]
    fn corotation_wind_at_pole() {
        let omega = 7.292115146706388e-5;
        let pos = DVec3::new(0.0, 0.0, 6_778_137.0); // north pole

        let wind = compute_corotation_wind(omega, pos);

        // omega × [0, 0, r] = [0, 0, 0] for Z-axis rotation
        assert!(wind.length() < 1e-10);
    }

    #[test]
    fn corotation_wind_zero_omega() {
        let pos = DVec3::new(7e6, 3e6, 1e6);
        let wind = compute_corotation_wind(0.0, pos);
        assert_eq!(wind, DVec3::ZERO);
    }

    #[test]
    fn typed_accessors_roundtrip_bit_identical() {
        let state = AtmosphereState::<Earth>::from_raw(
            1.225e-12,
            288.15,
            2.537e-10,
            DVec3::new(-359.7, 123.4, 0.5),
        );

        // Values extracted through typed accessors must equal the raw f64
        // inputs bit-for-bit — no unit conversion or round-trip loss.
        assert_eq!(
            state.density_typed().get::<kilogram_per_cubic_meter>(),
            state.density
        );
        assert_eq!(state.temperature_typed().get::<kelvin>(), state.temperature);
        assert_eq!(state.pressure_typed().get::<pascal>(), state.pressure);
        assert_eq!(state.wind.raw_si(), DVec3::new(-359.7, 123.4, 0.5));
    }

    #[test]
    fn typed_accessors_default_state() {
        let state = AtmosphereState::<Earth>::default();
        assert_eq!(state.density_typed().get::<kilogram_per_cubic_meter>(), 0.0);
        assert_eq!(state.temperature_typed().get::<kelvin>(), 0.0);
        assert_eq!(state.pressure_typed().get::<pascal>(), 0.0);
        assert_eq!(state.wind.raw_si(), DVec3::ZERO);
    }

    #[test]
    fn corotation_wind_typed_matches_raw() {
        let omega_raw = 7.292115146706388e-5;
        let pos_raw = DVec3::new(7.0e6, 3.0e6, 1.0e6);

        let typed_out = compute_corotation_wind_typed::<Earth>(
            omega_raw.rad_per_s(),
            pos_raw.m_at::<PlanetInertial<Earth>>(),
        );
        let raw_out = compute_corotation_wind(omega_raw, pos_raw);

        // Typed path must be bit-identical to the raw f64 path.
        assert_eq!(typed_out.raw_si(), raw_out);
    }

    #[test]
    fn corotation_wind_typed_zero_omega() {
        let pos = DVec3::new(7e6, 3e6, 1e6).m_at::<PlanetInertial<Earth>>();
        let wind = compute_corotation_wind_typed::<Earth>(0.0.rad_per_s(), pos);
        assert_eq!(wind.raw_si(), DVec3::ZERO);
    }

    #[test]
    fn relabel_self_planet_to_earth() {
        let state = AtmosphereState::<SelfPlanet>::from_raw(
            1.0e-12,
            300.0,
            1.0e-10,
            DVec3::new(10.0, 20.0, 0.0),
        );
        let earth = state.relabel::<Earth>();
        // Bit-identical SI values; only the phantom differs.
        assert_eq!(earth.density, 1.0e-12);
        assert_eq!(earth.temperature, 300.0);
        assert_eq!(earth.pressure, 1.0e-10);
        assert_eq!(earth.wind.raw_si(), DVec3::new(10.0, 20.0, 0.0));
    }
}
