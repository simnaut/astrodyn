//! Aerodynamic drag computation.
//!
//! Port of JEOD `models/interactions/aerodynamics/src/aero_drag.cc` and
//! `default_aero.cc` — the ballistic (default) drag model.
//!
//! In the free-molecular flow regime (orbital altitudes), drag force is:
//!   F = -0.5 · ρ · |v_rel|² · Cd · A · v̂_rel
//!
//! where v_rel is the vehicle velocity relative to the atmosphere (after
//! subtracting atmospheric wind/co-rotation).

use glam::{DMat3, DVec3};
use jeod_atmosphere::AtmosphereState;

/// Vehicle drag configuration for the ballistic (default) model.
///
/// Port of JEOD `DefaultAero` with `DRAG_OPT_CD` option.
#[derive(Debug, Clone, Copy)]
pub struct DragConfig {
    /// Coefficient of drag (dimensionless). Typically 2.0-2.5 for LEO.
    pub cd: f64,
    /// Cross-sectional area in m^2.
    pub area: f64,
    /// Override atmospheric density with a constant value (kg/m³).
    /// When `Some`, the atmosphere model's density is ignored and this
    /// value is used instead. Wind is still taken from the atmosphere.
    /// Port of JEOD `AerodynamicDrag::constant_density` + `density`.
    pub constant_density: Option<f64>,
}

/// Aerodynamic force and torque on a vehicle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AerodynamicForce {
    /// Aerodynamic force in N, in the structural/body frame.
    pub force: DVec3,
    /// Aerodynamic torque in N*m, in the structural/body frame.
    /// Zero for the ballistic model (force acts through center of mass).
    pub torque: DVec3,
}

impl Default for AerodynamicForce {
    fn default() -> Self {
        Self {
            force: DVec3::ZERO,
            torque: DVec3::ZERO,
        }
    }
}

/// Compute ballistic aerodynamic drag force.
///
/// Port of JEOD `AerodynamicDrag::aero_drag()` with `DefaultAero::DRAG_OPT_CD`.
///
/// # Arguments
/// * `config` - Vehicle drag properties (Cd, area)
/// * `atmos` - Atmospheric state at the vehicle (density, wind)
/// * `inertial_velocity` - Vehicle velocity in the inertial frame (m/s)
/// * `t_inertial_struct` - Rotation matrix: inertial -> structural/body frame
///
/// # Returns
/// Aerodynamic force and torque in the structural/body frame.
/// Torque is zero for the ballistic model.
pub fn compute_ballistic_drag(
    config: &DragConfig,
    atmos: &AtmosphereState,
    inertial_velocity: DVec3,
    t_inertial_struct: &DMat3,
) -> AerodynamicForce {
    // JEOD aero_drag.cc line 128: if(constant_density == false) { density = atmos_ptr->density; }
    let density = config.constant_density.unwrap_or(atmos.density);
    if density <= 0.0 {
        return AerodynamicForce::default();
    }

    // Relative velocity = vehicle velocity - atmospheric wind (in inertial frame)
    // JEOD aero_drag.cc line 111: Vector3::diff(inertial_velocity, atmos_ptr->wind, relative_vel_cm)
    let relative_vel_cm = inertial_velocity - atmos.wind;

    // Transform relative velocity to structural (body) frame
    // JEOD aero_drag.cc line 114: Vector3::transform(T_inertial_struct, relative_vel_cm, rel_vel_cm_struct)
    let rel_vel_cm_struct = *t_inertial_struct * relative_vel_cm;

    let rel_vel_mag = rel_vel_cm_struct.length();
    if rel_vel_mag < 1e-10 {
        return AerodynamicForce::default();
    }

    let rel_vel_struct_hat = rel_vel_cm_struct / rel_vel_mag;

    // Dynamic pressure: 0.5 · ρ · v²
    // JEOD aero_drag.cc line 132: param.dynamic_pressure = 0.5 * density * rel_vel_mag * rel_vel_mag
    let dynamic_pressure = 0.5 * density * rel_vel_mag * rel_vel_mag;

    // Drag force magnitude (negative = opposing motion)
    // JEOD default_aero.cc line 70: drag = -dynamic_pressure * area * Cd
    let drag = -dynamic_pressure * config.area * config.cd;

    // Force in structural frame: drag along relative velocity direction
    // JEOD default_aero.cc line 106: Vector3::scale(rel_vel_hat, drag, force)
    let force = rel_vel_struct_hat * drag;

    // Ballistic model: no torque (force acts through center of mass)
    // JEOD default_aero.cc line 107: Vector3::initialize(torque)
    AerodynamicForce {
        force,
        torque: DVec3::ZERO,
    }
}

/// Typed sibling of [`DragConfig`].
///
/// `cd` carries [`uom::si::f64::Ratio`] (dimensionless physical
/// quantity); `area` carries [`uom::si::f64::Area`] (m²);
/// `constant_density` (optional) carries [`uom::si::f64::MassDensity`]
/// (kg/m³).
#[derive(Debug, Clone, Copy)]
pub struct DragConfigTyped {
    pub cd: uom::si::f64::Ratio,
    pub area: uom::si::f64::Area,
    pub constant_density: Option<uom::si::f64::MassDensity>,
}

impl DragConfigTyped {
    /// Drop the dimensional annotations and emit the untyped storage form.
    /// Numeric values are preserved exactly (cd dimensionless, area in
    /// m², density in kg/m³).
    pub fn to_untyped(&self) -> DragConfig {
        DragConfig {
            cd: self.cd.value,
            area: self.area.value,
            constant_density: self.constant_density.map(|d| d.value),
        }
    }

    /// Wrap an untyped [`DragConfig`] as typed.
    pub fn from_untyped_unchecked(c: &DragConfig) -> Self {
        Self {
            cd: uom::si::f64::Ratio::new::<uom::si::ratio::ratio>(c.cd),
            area: uom::si::f64::Area::new::<uom::si::area::square_meter>(c.area),
            constant_density: c.constant_density.map(|d| {
                uom::si::f64::MassDensity::new::<uom::si::mass_density::kilogram_per_cubic_meter>(d)
            }),
        }
    }
}

/// Typed sibling of [`compute_ballistic_drag`].
///
/// Same numeric kernel — wraps [`DragConfigTyped`] / typed
/// [`Velocity<Inertial>`] inputs and unwraps to the existing
/// implementation. Output is **`Force<StructuralFrame<V>>`** because
/// [`compute_ballistic_drag`] returns the drag force in the vehicle's
/// structural/body frame (the frame `t_inertial_struct` rotates
/// *into*). Callers who need the force in the inertial integration
/// frame must rotate via the inverse `t_inertial_struct.transpose()`
/// — the typed signature makes the source frame explicit so this
/// conversion is a deliberate step rather than a silent assumption.
pub fn compute_ballistic_drag_typed<V: jeod_quantities::frame::Vehicle>(
    config: &DragConfigTyped,
    atmos: &AtmosphereState,
    inertial_velocity: jeod_quantities::aliases::Velocity<jeod_quantities::frame::Inertial>,
    t_inertial_struct: &DMat3,
) -> jeod_quantities::aliases::Force<jeod_quantities::frame::StructuralFrame<V>> {
    let untyped = compute_ballistic_drag(
        &config.to_untyped(),
        atmos,
        inertial_velocity.raw_si(),
        t_inertial_struct,
    );
    jeod_quantities::aliases::Force::<jeod_quantities::frame::StructuralFrame<V>>::from_raw_si(
        untyped.force,
    )
}

#[cfg(test)]
mod typed_tests {
    use super::*;
    use uom::si::area::square_meter;
    use uom::si::f64::{Area, MassDensity, Ratio};
    use uom::si::mass_density::kilogram_per_cubic_meter;
    use uom::si::ratio::ratio;

    #[test]
    fn drag_config_typed_round_trip() {
        let untyped = DragConfig {
            cd: 2.2,
            area: 4.5,
            constant_density: Some(1e-12),
        };
        let typed = DragConfigTyped::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();
        assert_eq!(back.cd, untyped.cd);
        assert_eq!(back.area, untyped.area);
        assert_eq!(back.constant_density, untyped.constant_density);
    }

    #[test]
    fn drag_config_typed_constant_density_none() {
        let untyped = DragConfig {
            cd: 2.0,
            area: 1.0,
            constant_density: None,
        };
        let typed = DragConfigTyped::from_untyped_unchecked(&untyped);
        assert!(typed.constant_density.is_none());
    }

    #[test]
    fn drag_config_typed_constructed_directly() {
        let typed = DragConfigTyped {
            cd: Ratio::new::<ratio>(2.2),
            area: Area::new::<square_meter>(4.5),
            constant_density: Some(MassDensity::new::<kilogram_per_cubic_meter>(1e-12)),
        };
        let untyped = typed.to_untyped();
        assert_eq!(untyped.cd, 2.2);
        assert_eq!(untyped.area, 4.5);
        assert_eq!(untyped.constant_density, Some(1e-12));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known drag force: F = 0.5 · ρ · v² · Cd · A
    #[test]
    fn known_drag_magnitude() {
        let config = DragConfig {
            cd: 2.2,
            area: 10.0, // m^2
            constant_density: None,
        };
        let density = 1e-12; // kg/m^3 (typical at 400 km)
        let velocity = 7600.0; // m/s (LEO orbital speed)

        let atmos = AtmosphereState {
            density,
            temperature: 0.0,
            pressure: 0.0,
            wind: DVec3::ZERO,
        };

        // Vehicle moving along +X in inertial, body aligned with inertial
        let vel = DVec3::new(velocity, 0.0, 0.0);
        let t = DMat3::IDENTITY;

        let result = compute_ballistic_drag(&config, &atmos, vel, &t);

        // Expected: F = 0.5 * 1e-12 * 7600^2 * 2.2 * 10 = 0.5 * 1e-12 * 5.776e7 * 22
        let expected_mag = 0.5 * density * velocity * velocity * config.cd * config.area;

        let force_mag = result.force.length();
        let rel_err = (force_mag - expected_mag).abs() / expected_mag;
        assert!(
            rel_err < 1e-12,
            "Drag magnitude: expected {expected_mag}, got {force_mag}"
        );

        // Force should be anti-velocity (negative X)
        assert!(result.force.x < 0.0, "Drag should oppose motion");
        assert!(result.force.y.abs() < 1e-20, "No Y component");
        assert!(result.force.z.abs() < 1e-20, "No Z component");
    }

    /// Zero density → zero drag.
    #[test]
    fn zero_density_zero_drag() {
        let config = DragConfig {
            cd: 2.2,
            area: 10.0,
            constant_density: None,
        };
        let atmos = AtmosphereState::default(); // density = 0
        let vel = DVec3::new(7600.0, 0.0, 0.0);

        let result = compute_ballistic_drag(&config, &atmos, vel, &DMat3::IDENTITY);
        assert_eq!(result.force, DVec3::ZERO);
    }

    /// Drag opposes relative velocity, not absolute velocity.
    #[test]
    fn drag_opposes_relative_velocity() {
        let config = DragConfig {
            cd: 2.0,
            area: 1.0,
            constant_density: None,
        };
        let atmos = AtmosphereState {
            density: 1e-12,
            temperature: 0.0,
            pressure: 0.0,
            wind: DVec3::new(100.0, 0.0, 0.0), // wind along +X
        };

        // Vehicle moving along +X at 7600 m/s
        let vel = DVec3::new(7600.0, 0.0, 0.0);
        let result = compute_ballistic_drag(&config, &atmos, vel, &DMat3::IDENTITY);

        // Relative velocity = 7600 - 100 = 7500 m/s along +X
        // Force should still oppose motion (negative X) but with reduced magnitude
        assert!(result.force.x < 0.0, "Drag should oppose relative velocity");

        // Compare with no-wind case
        let atmos_no_wind = AtmosphereState {
            wind: DVec3::ZERO,
            ..atmos
        };
        let result_no_wind = compute_ballistic_drag(&config, &atmos_no_wind, vel, &DMat3::IDENTITY);
        assert!(
            result.force.x.abs() < result_no_wind.force.x.abs(),
            "Wind reduces relative velocity, thus reduces drag"
        );
    }

    /// Torque is zero for ballistic model.
    #[test]
    fn ballistic_torque_is_zero() {
        let config = DragConfig {
            cd: 2.2,
            area: 10.0,
            constant_density: None,
        };
        let atmos = AtmosphereState {
            density: 1e-12,
            ..Default::default()
        };
        let vel = DVec3::new(7600.0, 0.0, 0.0);

        let result = compute_ballistic_drag(&config, &atmos, vel, &DMat3::IDENTITY);
        assert_eq!(result.torque, DVec3::ZERO);
    }

    /// Drag force is in the structural frame (rotated from inertial).
    #[test]
    fn drag_in_structural_frame() {
        let config = DragConfig {
            cd: 2.0,
            area: 1.0,
            constant_density: None,
        };
        let atmos = AtmosphereState {
            density: 1e-12,
            ..Default::default()
        };

        // Vehicle moving along +X in inertial
        let vel = DVec3::new(7600.0, 0.0, 0.0);

        // Body frame rotated 90° about Z: body X = inertial Y, body Y = -inertial X
        let t = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );

        let result = compute_ballistic_drag(&config, &atmos, vel, &t);

        // In body frame, velocity is along -Y (since body Y = -inertial X, wrong)
        // Actually: T * [7600, 0, 0] = [0*7600, 1*7600, 0] = [0, 7600, 0]
        // Wait, T transforms inertial to body:
        // body_vel = T * inertial_vel = [0*7600 + 1*0 + 0*0, -1*7600 + 0 + 0, 0]
        // Hmm, from_cols means column vectors. T * v:
        // T.col(0) * v.x + T.col(1) * v.y + T.col(2) * v.z
        // = [0, 1, 0] * 7600 + [-1, 0, 0] * 0 + [0, 0, 1] * 0
        // = [0, 7600, 0]
        // So in body frame, velocity is along +Y → drag force along -Y
        assert!(result.force.x.abs() < 1e-20, "No X component in body frame");
        assert!(result.force.y < 0.0, "Drag along -Y in body frame");
        assert!(result.force.z.abs() < 1e-20, "No Z component in body frame");
    }

    /// Order-of-magnitude check: ISS-like vehicle altitude loss.
    /// ISS: Cd*A/m ≈ 0.01 m²/kg, at 400 km, should lose ~100-300 m/day.
    #[test]
    fn iss_altitude_loss_order_of_magnitude() {
        let mass = 420_000.0; // kg
        let config = DragConfig {
            cd: 2.2,
            area: 1900.0, // m^2 (Cd*A/m ≈ 2.2*1900/420000 ≈ 0.01)
            constant_density: None,
        };

        // Typical density at 400 km during solar mean
        let density = 4e-12; // kg/m^3
        let atmos = AtmosphereState {
            density,
            ..Default::default()
        };

        let velocity = 7670.0; // m/s (ISS orbital speed)
        let vel = DVec3::new(velocity, 0.0, 0.0);

        let result = compute_ballistic_drag(&config, &atmos, vel, &DMat3::IDENTITY);
        let drag_accel = result.force.length() / mass; // m/s^2

        // Semi-major axis decay rate: da/dt ≈ -2*a*drag_accel/v (approximate)
        let a = 6_778_000.0; // m (400 km altitude)
        let da_dt = 2.0 * a * drag_accel / velocity; // m/s
        let da_day = da_dt * 86400.0; // m/day

        assert!(
            da_day > 50.0 && da_day < 1000.0,
            "ISS altitude loss should be ~100-300 m/day, got {} m/day",
            da_day
        );
    }
}
