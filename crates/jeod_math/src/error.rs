#[derive(Debug, thiserror::Error)]
pub enum OrbitalError {
    #[error("Kepler equation failed to converge after {0} iterations")]
    KeplerConvergence(usize),
    #[error("Invalid gravitational parameter: {0}")]
    InvalidMu(f64),
    #[error("Degenerate orbit: zero position or velocity magnitude")]
    DegenerateOrbit,
}
