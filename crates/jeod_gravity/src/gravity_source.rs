use crate::spherical_harmonics_gravity_source::SphericalHarmonicsData;

#[derive(Debug, Clone)]
pub struct GravitySource {
    pub mu: f64, // gravitational parameter, m^3/s^2
    pub model: GravityModel,
}

#[derive(Debug, Clone)]
pub enum GravityModel {
    PointMass,
    SphericalHarmonics(Box<SphericalHarmonicsData>),
}
