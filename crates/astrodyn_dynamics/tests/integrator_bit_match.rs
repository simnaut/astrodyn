//! Bit-identity regression test for the RK4 Kepler propagation path.
//!
//! Issue #101's type-system refactor wraps f64 computations in uom/typed
//! newtypes. We expect these wrappers to be zero-cost — the compiler's
//! IR should be byte-identical pre/post wrapping. This test pins a SHA-256
//! of the integrator's output to catch any reordering that would shift
//! downstream Tier 3 baselines.
//!
//! If this test fails after a refactor-only commit, investigate *before*
//! refreezing the hash: the Tier 3 baselines.json is downstream of this.

use astrodyn_dynamics::{rk4_translational_step, TranslationalState};
use glam::DVec3;
use sha2::{Digest, Sha256};

/// Earth gravitational parameter (m^3/s^2), same literal used elsewhere in the
/// workspace (e.g. `examples/kepler_orbit.rs`).
const MU_EARTH: f64 = 3.986_004_415e14;

/// Number of RK4 steps to take. 1000 steps at 60 s each = 60000 s ≈ 2.9 ISS
/// orbits, exercising many integration rounds without being test-time costly.
const NUM_STEPS: usize = 1000;

/// Fixed integration step size (seconds).
const DT: f64 = 60.0;

/// Pinned SHA-256 of the canonical byte representation of the final state.
/// See the module docstring: update only after verifying that an IR change is
/// intentional.
const PINNED_HASH: &str = "7b87ea77d0f00400664c6f9711375362e050346ad5a086fa7db59b44e0f0b037";

/// Point-mass Earth gravity acceleration (two-body Kepler problem).
fn kepler_accel(state: &TranslationalState, _time_frac: f64) -> DVec3 {
    let r = state.position.length();
    -MU_EARTH / (r * r * r) * state.position
}

/// Format a single f64 as a lossless decimal string. The `{:.17e}` format
/// guarantees round-trip equality for finite f64 values, so the hash depends
/// only on the numerical value, not on endianness or host-architecture
/// in-memory layout.
fn fmt(v: f64) -> String {
    format!("{:.17e}", v)
}

#[test]
fn rk4_kepler_bit_match() {
    // ISS-like circular initial conditions (SI units). Chosen to match the
    // profile used in `examples/kepler_orbit.rs`: a ~400 km altitude equatorial
    // orbit.
    let mut state = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7_668.589_305_449_398, 0.0),
    };

    for _ in 0..NUM_STEPS {
        state = rk4_translational_step(&state, kepler_accel, DT);
    }

    // Canonical byte representation: six f64 components in fixed order,
    // each formatted as `{:.17e}` and newline-separated (trailing newline).
    let mut canonical = String::with_capacity(6 * 25);
    canonical.push_str(&fmt(state.position.x));
    canonical.push('\n');
    canonical.push_str(&fmt(state.position.y));
    canonical.push('\n');
    canonical.push_str(&fmt(state.position.z));
    canonical.push('\n');
    canonical.push_str(&fmt(state.velocity.x));
    canonical.push('\n');
    canonical.push_str(&fmt(state.velocity.y));
    canonical.push('\n');
    canonical.push_str(&fmt(state.velocity.z));
    canonical.push('\n');

    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let hash_bytes = hasher.finalize();
    let hash_hex: String = hash_bytes.iter().map(|b| format!("{b:02x}")).collect();

    assert_eq!(
        hash_hex, PINNED_HASH,
        "RK4 Kepler bit-identity regression: the integrator output changed.\n\
         Canonical bytes follow — investigate *before* refreezing the hash.\n\
         ----- BEGIN CANONICAL BYTES -----\n\
         {canonical}\
         ----- END CANONICAL BYTES -----",
    );
}
