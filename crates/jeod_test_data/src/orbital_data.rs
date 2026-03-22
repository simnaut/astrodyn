use glam::DVec3;
use regex::Regex;

/// A Cartesian state vector (position + velocity) from JEOD's orbital elements
/// verification data.
#[derive(Debug, Clone)]
pub struct CartesianStateVector {
    pub position: DVec3, // meters
    pub velocity: DVec3, // m/s
}

/// Load orbital element test vectors from JEOD's `orb_ell_in.py`.
///
/// The file contains 5001 Cartesian state vectors with lines like:
/// ```python
/// orb_elem_test.orb_elem_multi_ver.data_vals_in[0]  = [ 4.08e+06, 4.28e+06, 1.66e+06, -5.80e+03, 3.30e+03, 5.76e+03]
/// ```
///
/// Each line has 6 values: position (x,y,z) in meters, velocity (x,y,z) in m/s.
///
/// # Panics
/// Panics if the file cannot be read.
pub fn load_orbital_test_vectors(jeod_root: &std::path::Path) -> Vec<CartesianStateVector> {
    let path = jeod_root.join(
        "models/utils/orbital_elements/verif/SIM_orb_elem/Modified_data/orb_ell_in.py",
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    let array_re = Regex::new(
        r"\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]",
    )
    .unwrap();

    let mut vectors = Vec::new();

    for cap in array_re.captures_iter(&content) {
        let px: f64 = cap[1].parse().unwrap();
        let py: f64 = cap[2].parse().unwrap();
        let pz: f64 = cap[3].parse().unwrap();
        let vx: f64 = cap[4].parse().unwrap();
        let vy: f64 = cap[5].parse().unwrap();
        let vz: f64 = cap[6].parse().unwrap();

        vectors.push(CartesianStateVector {
            position: DVec3::new(px, py, pz),
            velocity: DVec3::new(vx, vy, vz),
        });
    }

    vectors
}
