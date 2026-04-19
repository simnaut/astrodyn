use crate::epoch::TAI_TT_OFFSET;
use crate::{SecondsSince, TAI, TT};

/// Convert TAI seconds-since-epoch to TT seconds-since-epoch.
/// TT = TAI + 32.184s (exact by definition).
pub fn tai_to_tt(tai_seconds: f64) -> f64 {
    tai_seconds + TAI_TT_OFFSET
}

/// Convert TT seconds-since-epoch to TAI seconds-since-epoch.
pub fn tt_to_tai(tt_seconds: f64) -> f64 {
    tt_seconds - TAI_TT_OFFSET
}

// --- Typed variants (Phase 1 #103) ------------------------------------------
//
// Additive typed siblings. They forward to the f64 free functions above so
// behavior is bit-identical — never routed through
// `TimeConverter<TAI, TT>::apply()`, which would re-round through
// `from_seconds`/`as_seconds`.

/// Typed TAI → TT converter (delegates to [`tai_to_tt`]).
pub fn tai_to_tt_typed(t: SecondsSince<TAI>) -> SecondsSince<TT> {
    SecondsSince::from_seconds(tai_to_tt(t.as_seconds()))
}

/// Typed TT → TAI converter (delegates to [`tt_to_tai`]).
pub fn tt_to_tai_typed(t: SecondsSince<TT>) -> SecondsSince<TAI> {
    SecondsSince::from_seconds(tt_to_tai(t.as_seconds()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tai_tt_exact_offset() {
        assert_eq!(tai_to_tt(0.0), 32.184);
        assert_eq!(tai_to_tt(100.0), 132.184);
        assert_eq!(tt_to_tai(32.184), 0.0);
    }

    #[test]
    fn tai_tt_round_trip() {
        let tai = 123456.789;
        let tt = tai_to_tt(tai);
        let back = tt_to_tai(tt);
        assert!((back - tai).abs() < 1e-15);
    }

    #[test]
    fn tai_tt_typed_matches_f64() {
        // Typed wrappers must agree with the f64 versions bit-exactly.
        let tai_raw = 1_000_000.0_f64;
        let tt_typed = tai_to_tt_typed(SecondsSince::<TAI>::from_seconds(tai_raw));
        assert_eq!(tt_typed.as_seconds(), tai_to_tt(tai_raw));
    }

    #[test]
    fn tai_tt_typed_round_trip() {
        // Non-trivial input; tolerance 1e-14 s (addition/subtraction of
        // TAI_TT_OFFSET at this magnitude is exact in f64, but we still
        // allow 1e-14 per the Phase 1 spec).
        let tai = SecondsSince::<TAI>::from_seconds(1_000_000.0);
        let tt = tai_to_tt_typed(tai);
        let back = tt_to_tai_typed(tt);
        let err = (back.as_seconds() - tai.as_seconds()).abs();
        assert!(err < 1e-14, "round-trip err={} (tolerance 1e-14 s)", err);
    }
}
