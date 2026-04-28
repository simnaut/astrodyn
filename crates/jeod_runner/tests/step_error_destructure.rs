//! Compile-time witness that `StepError` variant fields are part of the
//! public surface and can be destructured from a downstream crate.
//!
//! Background: a review of #172 L2 raised the concern that struct-variant
//! fields (`source_idx`, `tdb_jd`, `body_idx`, …) might be inaccessible
//! to downstream recovery logic. In Rust, an enum variant's fields inherit
//! the enum's visibility, so when the enum is `pub` the fields are part
//! of the public surface and `pub field: T` syntax is neither needed nor
//! accepted. This integration test proves the pattern by destructuring
//! both variants from outside `jeod_runner::error` (specifically, from a
//! `tests/` integration target, which sits in the same crate-as-consumer
//! position downstream code does).
//!
//! The `error_path_executes_known_unwrap` smoke test additionally
//! confirms `StepError` round-trips through the `Display` /
//! `core::error::Error` traits without panicking.

use jeod_runner::StepError;
use jeod_sim::EphemerisBody;

#[test]
fn ephemeris_lookup_destructures_all_fields() {
    let err = StepError::EphemerisLookup {
        source_idx: 7,
        target: EphemerisBody::Sun,
        observer: EphemerisBody::Earth,
        tdb_jd: 2_460_000.5,
        message: "test failure".to_string(),
    };
    match err {
        StepError::EphemerisLookup {
            source_idx,
            target,
            observer,
            tdb_jd,
            message,
        } => {
            assert_eq!(source_idx, 7);
            assert_eq!(target, EphemerisBody::Sun);
            assert_eq!(observer, EphemerisBody::Earth);
            assert!((tdb_jd - 2_460_000.5).abs() < 1e-9);
            assert_eq!(message, "test failure");
        }
        StepError::FrameSwitchTargetMissing { .. } => {
            panic!("matched the wrong variant")
        }
    }
}

#[test]
fn frame_switch_target_missing_destructures_all_fields() {
    let err = StepError::FrameSwitchTargetMissing {
        body_idx: 3,
        target_source: 99,
        num_sources: 4,
    };
    match err {
        StepError::FrameSwitchTargetMissing {
            body_idx,
            target_source,
            num_sources,
        } => {
            assert_eq!(body_idx, 3);
            assert_eq!(target_source, 99);
            assert_eq!(num_sources, 4);
        }
        StepError::EphemerisLookup { .. } => panic!("matched the wrong variant"),
    }
}

#[test]
fn step_error_implements_core_error() {
    let err = StepError::FrameSwitchTargetMissing {
        body_idx: 0,
        target_source: 0,
        num_sources: 0,
    };
    // Smoke check: the trait composition (Display + core::error::Error)
    // works through trait objects, which is what `?` and downstream
    // error wrappers rely on.
    let dyn_err: &dyn core::error::Error = &err;
    let msg = format!("{dyn_err}");
    assert!(msg.contains("frame switch evaluation"));
}
