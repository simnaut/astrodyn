//! Replay series (v1): segment-per-topology-change recording.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::document::{validate_header, validate_record, DocError, DocHeader, FrameRecord};
use crate::FrameUid;

/// All frame records at one simulation instant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochRow {
    /// Elapsed simulation seconds of this row.
    pub simtime: f64,
    /// One record per frame node; each names its parent uid.
    pub records: Vec<FrameRecord>,
}

/// A maximal run of epochs sharing one topology. Any topology change (a
/// frame-switch reparent, an attach) closes the segment — boundaries
/// double as seek keyframes for replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameSegment {
    /// Simtime of the segment's first epoch.
    pub start_simtime: f64,
    /// Epoch rows in recording order.
    pub epochs: Vec<EpochRow>,
}

/// A recorded run: header + interned uid table + topology-stable segments
/// (replay v1; streaming topology events are v2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameSeries {
    /// Schema version, conventions, time anchor (simtime of the *first*
    /// recorded epoch).
    pub header: DocHeader,
    /// Interned identity table shared by every record in every segment.
    pub uids: Vec<FrameUid>,
    /// Topology-stable segments in time order.
    pub segments: Vec<FrameSegment>,
}

impl FrameSeries {
    /// Validate the header (before interpreting any state), the uid
    /// table, every record, and the replay-v1 structural invariants:
    ///
    /// - **within one segment the declared topology is constant** — a row
    ///   whose parent assignments differ from the segment's first row is
    ///   a recording bug ([`SeriesBuilder`] can never produce one);
    /// - **every epoch row covers the full uid table** (fixed population:
    ///   one record per frame per epoch — a partial row would make
    ///   consumers silently operate on incomplete state);
    /// - **segments are non-empty** and their `start_simtime` equals
    ///   their first epoch's `simtime` (the boundary is the seek
    ///   keyframe).
    pub fn validate(&self) -> Result<(), DocError> {
        validate_header(&self.header)?;
        crate::document::validate_uid_table(&self.uids)?;
        for (seg_pos, seg) in self.segments.iter().enumerate() {
            let Some(first) = seg.epochs.first() else {
                return Err(DocError::EmptySegment { segment: seg_pos });
            };
            if seg.start_simtime.to_bits() != first.simtime.to_bits() {
                return Err(DocError::SegmentStartMismatch {
                    segment: seg_pos,
                    start: seg.start_simtime,
                    first_epoch: first.simtime,
                });
            }
            let mut segment_topology: Option<BTreeMap<u32, Option<u32>>> = None;
            for row in &seg.epochs {
                if !row.simtime.is_finite() {
                    return Err(DocError::NonFinite(format!(
                        "segment {seg_pos} epoch simtime = {}",
                        row.simtime
                    )));
                }
                if row.records.len() != self.uids.len() {
                    return Err(DocError::IncompleteRow {
                        segment: seg_pos,
                        simtime: row.simtime,
                        found: row.records.len(),
                        expected: self.uids.len(),
                    });
                }
                let mut seen = vec![false; self.uids.len()];
                for (i, rec) in row.records.iter().enumerate() {
                    validate_record(rec, i, self.uids.len())?;
                    let idx = rec.uid_index as usize;
                    if seen[idx] {
                        return Err(DocError::DuplicateUid {
                            index: rec.uid_index,
                        });
                    }
                    seen[idx] = true;
                }
                let topology = fold_topology(&row.records);
                match &segment_topology {
                    None => segment_topology = Some(topology),
                    Some(expected) => {
                        if &topology != expected {
                            let uid_index = differing_uid(expected, &topology);
                            return Err(DocError::TopologyMismatch {
                                segment: seg_pos,
                                simtime: row.simtime,
                                uid_index,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Serialize to a JSON string.
    ///
    /// # Panics
    /// Panics if the series fails [`Self::validate`] (non-finite values,
    /// index errors, or an intra-segment topology change).
    pub fn to_json_string(&self) -> String {
        self.validate().unwrap_or_else(|err| {
            panic!("FrameSeries::to_json_string: refusing to serialize an invalid series: {err}")
        });
        serde_json::to_string(self)
            .expect("FrameSeries serialization is infallible after validate()")
    }

    /// Parse from a JSON string and [`Self::validate`].
    pub fn from_json_str(json: &str) -> Result<Self, DocError> {
        let series: Self = serde_json::from_str(json)?;
        series.validate()?;
        Ok(series)
    }
}

/// Fold one row's declared topology: uid index → declared parent uid index.
fn fold_topology(records: &[FrameRecord]) -> BTreeMap<u32, Option<u32>> {
    records.iter().map(|r| (r.uid_index, r.parent)).collect()
}

/// First uid index whose parent assignment differs between two folded
/// topologies (also covers a frame appearing/disappearing).
fn differing_uid(a: &BTreeMap<u32, Option<u32>>, b: &BTreeMap<u32, Option<u32>>) -> u32 {
    for (uid, parent) in a {
        if b.get(uid) != Some(parent) {
            return *uid;
        }
    }
    for uid in b.keys() {
        if !a.contains_key(uid) {
            return *uid;
        }
    }
    // Caller only invokes this when the maps differ.
    unreachable!("differing_uid called on identical topologies")
}

/// Writer-side recorder: push one epoch's rows per step; the builder
/// auto-closes the open segment whenever the **declared topology** (each
/// record's parent uid) changes, so producers never hand-manage segment
/// boundaries and an intra-segment topology change is unrepresentable.
#[derive(Debug)]
pub struct SeriesBuilder {
    header: DocHeader,
    uids: Vec<FrameUid>,
    segments: Vec<FrameSegment>,
    open: Option<(FrameSegment, BTreeMap<u32, Option<u32>>)>,
}

impl SeriesBuilder {
    /// Start a series. `uids` is the interned identity table every pushed
    /// record indexes into.
    pub fn new(header: DocHeader, uids: Vec<FrameUid>) -> Self {
        Self {
            header,
            uids,
            segments: Vec::new(),
            open: None,
        }
    }

    /// The interned identity table (for producers building records).
    pub fn uids(&self) -> &[FrameUid] {
        &self.uids
    }

    /// Append one epoch's records. Opens a new segment when the declared
    /// topology differs from the open segment's (or on the first push).
    pub fn push_epoch(&mut self, simtime: f64, records: Vec<FrameRecord>) {
        assert!(
            simtime.is_finite(),
            "SeriesBuilder::push_epoch: simtime is non-finite ({simtime}) — fix the \
             producing simulation clock"
        );
        let topology = fold_topology(&records);
        let row = EpochRow { simtime, records };
        match &mut self.open {
            Some((segment, open_topology)) if *open_topology == topology => {
                segment.epochs.push(row);
            }
            _ => {
                // Topology changed (or first epoch): close the open
                // segment and start a new one at this row — the boundary
                // is the seek keyframe.
                if let Some((segment, _)) = self.open.take() {
                    self.segments.push(segment);
                }
                self.open = Some((
                    FrameSegment {
                        start_simtime: simtime,
                        epochs: vec![row],
                    },
                    topology,
                ));
            }
        }
    }

    /// Finish recording and return the series.
    pub fn finish(mut self) -> FrameSeries {
        if let Some((segment, _)) = self.open.take() {
            self.segments.push(segment);
        }
        FrameSeries {
            header: self.header,
            uids: self.uids,
            segments: self.segments,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{
        body_record, header, pfix_record, record_bits, root_record, uid_table,
    };

    fn row() -> Vec<FrameRecord> {
        vec![root_record(), pfix_record(), body_record()]
    }

    /// The same row with the body reparented under the pfix frame —
    /// the wire-level shape of a frame-switch reparent.
    fn reparented_row() -> Vec<FrameRecord> {
        let mut records = row();
        records[2].parent = Some(1);
        records
    }

    #[test]
    fn series_builder_single_segment_when_topology_stable() {
        let mut builder = SeriesBuilder::new(header(), uid_table());
        for i in 0..5 {
            builder.push_epoch(f64::from(i), row());
        }
        let series = builder.finish();
        series.validate().expect("stable-topology series validates");
        assert_eq!(series.segments.len(), 1);
        assert_eq!(series.segments[0].epochs.len(), 5);
        assert_eq!(
            series.segments[0].start_simtime.to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn series_builder_opens_new_segment_on_reparent() {
        let mut builder = SeriesBuilder::new(header(), uid_table());
        builder.push_epoch(0.0, row());
        builder.push_epoch(1.0, row());
        builder.push_epoch(2.0, reparented_row()); // frame switch fired
        builder.push_epoch(3.0, reparented_row());
        let series = builder.finish();
        series.validate().expect("segmented series validates");
        assert_eq!(series.segments.len(), 2, "reparent must close the segment");
        assert_eq!(series.segments[0].epochs.len(), 2);
        assert_eq!(series.segments[1].epochs.len(), 2);
        // The boundary doubles as a seek keyframe: the new segment starts
        // at the reparent epoch.
        assert_eq!(
            series.segments[1].start_simtime.to_bits(),
            2.0_f64.to_bits()
        );
    }

    #[test]
    fn series_round_trips_bit_exact() {
        let mut builder = SeriesBuilder::new(header(), uid_table());
        builder.push_epoch(0.5, row());
        builder.push_epoch(1.5, reparented_row());
        let series = builder.finish();
        let json = series.to_json_string();
        let back = FrameSeries::from_json_str(&json).expect("round trip");
        assert_eq!(back.segments.len(), series.segments.len());
        for (sa, sb) in series.segments.iter().zip(&back.segments) {
            assert_eq!(sa.start_simtime.to_bits(), sb.start_simtime.to_bits());
            for (ra, rb) in sa.epochs.iter().zip(&sb.epochs) {
                assert_eq!(ra.simtime.to_bits(), rb.simtime.to_bits());
                for (a, b) in ra.records.iter().zip(&rb.records) {
                    assert_eq!(record_bits(a), record_bits(b));
                }
            }
        }
    }

    #[test]
    fn validate_rejects_intra_segment_topology_change() {
        // Hand-build the malformed series the SeriesBuilder can never
        // produce: a reparent INSIDE one segment.
        let series = FrameSeries {
            header: header(),
            uids: uid_table(),
            segments: vec![FrameSegment {
                start_simtime: 0.0,
                epochs: vec![
                    EpochRow {
                        simtime: 0.0,
                        records: row(),
                    },
                    EpochRow {
                        simtime: 1.0,
                        records: reparented_row(),
                    },
                ],
            }],
        };
        assert!(matches!(
            series.validate(),
            Err(DocError::TopologyMismatch {
                segment: 0,
                uid_index: 2,
                ..
            })
        ));
    }

    #[test]
    #[should_panic(expected = "simtime is non-finite")]
    fn push_epoch_rejects_non_finite_simtime() {
        let mut builder = SeriesBuilder::new(header(), uid_table());
        builder.push_epoch(f64::NAN, row());
    }

    #[test]
    fn validate_rejects_incomplete_row() {
        // A row that consistently omits a frame would make consumers
        // silently operate on partial state — replay v1 is
        // fixed-population, every row covers the full uid table.
        let mut builder = SeriesBuilder::new(header(), uid_table());
        builder.push_epoch(0.0, row());
        let mut series = builder.finish();
        series.segments[0].epochs[0].records.pop();
        assert!(matches!(
            series.validate(),
            Err(DocError::IncompleteRow {
                segment: 0,
                found: 2,
                expected: 3,
                ..
            })
        ));
    }

    #[test]
    fn validate_rejects_empty_segment() {
        let series = FrameSeries {
            header: header(),
            uids: uid_table(),
            segments: vec![FrameSegment {
                start_simtime: 0.0,
                epochs: vec![],
            }],
        };
        assert!(matches!(
            series.validate(),
            Err(DocError::EmptySegment { segment: 0 })
        ));
    }

    #[test]
    fn validate_rejects_segment_start_mismatch() {
        // The boundary doubles as the seek keyframe — stale seek
        // metadata is silently wrong replay.
        let mut builder = SeriesBuilder::new(header(), uid_table());
        builder.push_epoch(1.0, row());
        let mut series = builder.finish();
        series.segments[0].start_simtime = 0.5;
        assert!(matches!(
            series.validate(),
            Err(DocError::SegmentStartMismatch { segment: 0, .. })
        ));
    }
}
