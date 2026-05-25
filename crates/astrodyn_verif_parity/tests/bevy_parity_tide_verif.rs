//! Bevy ↔ runner parity for SIM_tide_verif (solid-body tidal ΔC20 with
//! per-step Sun + Moon DE421 position injection), via the
//! `VerificationCaseParityExt` trait.
//!
//! Unblocked by issue #395's `BevySimContext`: the recipe's `pre_step`
//! drives `set_source_position` (Sun/Moon) and `set_tidal_body_position`
//! (Sun/Moon inside Earth's tidal config) on both runtimes. Bit-identity
//! holds because both sides receive the same numeric inputs at every
//! reference-CSV record before integration runs.
//!
//! The corresponding runner-vs-JEOD test is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_tide_verif.rs`.

use astrodyn_verif_jeod::run_verification::sim_tide_verif;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_tide_verif_run01() {
    sim_tide_verif::run01().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_tide_verif_run02() {
    sim_tide_verif::run02().run_and_assert_parity::<astrodyn::Earth>();
}
