use crate::epoch::TAI_TT_OFFSET;

/// Convert TAI seconds-since-epoch to TT seconds-since-epoch.
/// TT = TAI + 32.184s (exact by definition).
pub fn tai_to_tt(tai_seconds: f64) -> f64 {
    tai_seconds + TAI_TT_OFFSET
}

/// Convert TT seconds-since-epoch to TAI seconds-since-epoch.
pub fn tt_to_tai(tt_seconds: f64) -> f64 {
    tt_seconds - TAI_TT_OFFSET
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
}
