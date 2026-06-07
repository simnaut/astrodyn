//! Snapshot document types: header, conventions, records, validation.

use astrodyn_quantities::frame_descriptor::FrameUid;
use serde::{Deserialize, Serialize};

/// Version of the wire schema this build writes and accepts.
///
/// Bump on any change a v1 reader cannot interpret; additive, optional
/// fields do not require a bump.
pub const SCHEMA_VERSION: u32 = 1;

/// Numeric conventions carried **in-band** and validated before any state
/// is interpreted ([`FrameDocument::validate`]). Self-describing strings
/// rather than flags so a code-free reader (RFS-602) sees the convention,
/// not a boolean it must look up.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conventions {
    /// Translation convention (JEOD_INV RF.06).
    pub translation: String,
    /// Rotation convention (JEOD_INV RF.07 storage form; RF.04 canonicity).
    pub rotation: String,
    /// Angular-velocity convention.
    pub angular_velocity: String,
    /// Time scale of `simtime` and per-record epochs.
    pub time_scale: String,
}

impl Conventions {
    /// The conventions this build writes. Loading a document whose
    /// conventions differ fails before any state is interpreted.
    pub fn current() -> Self {
        Self {
            translation: "position/velocity in parent-frame coordinates, SI (m, m/s)".into(),
            rotation: "scalar-first left-transformation quaternion parent->this; \
                       matrix is the same transformation; the non-canonical \
                       representation is re-derived on load"
                .into(),
            angular_velocity: "this-frame coordinates, rad/s".into(),
            time_scale: "TDB seconds since J2000 epoch".into(),
        }
    }
}

/// Document header: schema version, conventions, and the producing
/// simulation's time anchor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocHeader {
    /// Wire-schema version ([`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// In-band numeric conventions, validated before interpreting state.
    pub conventions: Conventions,
    /// Elapsed simulation seconds at the snapshot (exact `f64`; a restoring
    /// host advances a fresh simulation by this value in one step, which is
    /// bit-exact: `0.0 + simtime` is a single addition).
    pub simtime: f64,
    /// The producing simulation's TAI truncated-Julian epoch. A restoring
    /// host must be configured with the **same** epoch — derived time
    /// scales (TDB, GMST) are functions of it, so applying a document to a
    /// simulation at a different epoch is loudly rejected, never silently
    /// reinterpreted.
    pub tai_tjt_at_epoch: f64,
}

/// Rotation state in whichever representation was **canonical at the write
/// site** (RFS-601: serialize the canonical field; re-derive the cached
/// form on load).
///
/// - The typed construction path is quaternion-canonical (JEOD_INV RF.04:
///   the matrix is derived from the normalized quaternion).
/// - Rotation-model writers (`sync_pfix_rotation`) are matrix-canonical:
///   the RNP/IAU matrix is stored verbatim and the quaternion derived.
///
/// Serializing the non-canonical form would lose bits through the
/// conversion round trip; carrying the canonical form makes
/// serialize → reload → continue bit-identical for both regimes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalRotation {
    /// Scalar-first left-transformation quaternion `[q0, q1, q2, q3]`
    /// (parent → this; JEOD_INV RF.07).
    Quat([f64; 4]),
    /// 3×3 transformation matrix (parent → this) as **columns**
    /// `[col0, col1, col2]` — the producer's native column-major layout,
    /// carried losslessly.
    Matrix([[f64; 3]; 3]),
}

/// Translational state relative to the parent frame (JEOD_INV RF.06).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransRecord {
    /// Position in parent-frame coordinates (m).
    pub position: [f64; 3],
    /// Velocity in parent-frame coordinates (m/s).
    pub velocity: [f64; 3],
}

/// Where a record's state comes from (RFS-603) — the three production
/// write regimes, distinguishable by consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The state is a projection of an authoritative body store (the tree
    /// node is dual-written per step). The payload carries the store's
    /// rotational half verbatim: the body **node**'s rotation is stale by
    /// design (per-step writeback syncs translation only), so node fields
    /// and store fields genuinely differ and both are needed for a
    /// bit-exact restore. The translational half is *not* duplicated —
    /// node and store translations are bit-identical by dual-write
    /// construction.
    Integrated {
        /// Body-store attitude quaternion, scalar-first `[q0, q1, q2, q3]`
        /// (inertial → body). `None` for 3-DOF bodies.
        attitude_quat: Option<[f64; 4]>,
        /// Body-store angular velocity in body-frame coordinates (rad/s).
        /// `None` for 3-DOF bodies.
        ang_vel_body: Option<[f64; 3]>,
    },
    /// The state is an evaluation of `(model, epoch)` — an ephemeris query
    /// or a rotation-model composition. Re-derivable: a restoring host
    /// recomputes it from the model at the restored time; the materialized
    /// values are carried for code-free consumers.
    Derived {
        /// Producer-meaningful model identifier (e.g. `"EarthRNP"`,
        /// `"DE4xx:Sun/Earth"`).
        model: String,
    },
    /// Caller-supplied ground truth (static configuration or an explicit
    /// retarget). Not re-derivable; the materialized values are the truth.
    Injected,
}

/// One frame node's serialized state.
///
/// `uid_index` and `parent` index into the document's interned
/// [`FrameUid`] table — **every record names the parent uid it is relative
/// to** (`None` = root), so consumers can check folded topology per record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRecord {
    /// Diagnostic node label (identity lives in the uid, not here).
    pub name: String,
    /// Index of this frame's identity in the document's uid table.
    pub uid_index: u32,
    /// Index of the **parent frame's identity** in the uid table; `None`
    /// for a root.
    pub parent: Option<u32>,
    /// Frame epoch: the time-validity of this state, TDB seconds (RFS-603).
    /// `None` only for nodes never stamped by a per-step write.
    pub epoch: Option<f64>,
    /// Translational state relative to the parent.
    pub trans: TransRecord,
    /// Rotational state in its canonical representation.
    pub rotation: CanonicalRotation,
    /// Angular velocity of this frame relative to parent, this-frame
    /// coordinates (rad/s).
    pub ang_vel_this: [f64; 3],
    /// Where this state comes from.
    pub origin: Origin,
}

/// A frame-tree snapshot: header + interned uid table + one record per
/// node. Records appear in producer node order; `parent` references are by
/// uid-table index, never by arena index (RFS-601: identity is stable
/// across the wire, storage indices are not).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameDocument {
    /// Schema version, conventions, time anchor.
    pub header: DocHeader,
    /// Interned identity table; records reference it by index.
    pub uids: Vec<FrameUid>,
    /// One record per frame node.
    pub records: Vec<FrameRecord>,
}

/// Validation / load errors for documents and series.
#[derive(Debug, thiserror::Error)]
pub enum DocError {
    /// The document was written by an incompatible schema version.
    #[error(
        "unsupported frame-document schema version {found}; this build \
         supports version {SCHEMA_VERSION}"
    )]
    UnsupportedVersion {
        /// Version found in the header.
        found: u32,
    },
    /// A header convention differs from this build's conventions — the
    /// state cannot be interpreted.
    #[error(
        "frame-document convention mismatch in `{field}`: document says \
         {found:?}, this build expects {expected:?} — refusing to interpret \
         state under a different convention"
    )]
    ConventionMismatch {
        /// Which convention field disagrees.
        field: &'static str,
        /// The document's value.
        found: String,
        /// This build's value.
        expected: String,
    },
    /// A record's `uid_index` is out of range for the uid table.
    #[error("record {record} names uid index {index}, but the table has {len} entries")]
    UidIndexOutOfRange {
        /// Record position.
        record: usize,
        /// Offending index.
        index: u32,
        /// Table length.
        len: usize,
    },
    /// A record's `parent` is out of range for the uid table.
    #[error("record {record} names parent uid index {index}, but the table has {len} entries")]
    ParentIndexOutOfRange {
        /// Record position.
        record: usize,
        /// Offending index.
        index: u32,
        /// Table length.
        len: usize,
    },
    /// A record names itself as its parent.
    #[error("record {record} (uid index {index}) names itself as its parent")]
    SelfParent {
        /// Record position.
        record: usize,
        /// The record's uid index.
        index: u32,
    },
    /// A numeric field is NaN or infinite. A non-finite value in a frame
    /// document is upstream broken physics — fix the producer.
    #[error("non-finite value in {0} — fix the producing physics before serializing")]
    NonFinite(String),
    /// The same uid appears on more than one record of a snapshot (or of
    /// one epoch row in a series).
    #[error("uid index {index} appears on more than one record")]
    DuplicateUid {
        /// The duplicated uid-table index.
        index: u32,
    },
    /// In a series, an epoch row inside one segment declares a different
    /// topology than the segment's first row — topology changes must close
    /// the segment (replay v1: segment-per-topology-change).
    #[error(
        "segment {segment} epoch at simtime {simtime} declares a parent for uid \
         index {uid_index} that differs from the segment's topology — a \
         topology change must open a new segment"
    )]
    TopologyMismatch {
        /// Segment position.
        segment: usize,
        /// Offending epoch's simtime.
        simtime: f64,
        /// The frame whose declared parent changed.
        uid_index: u32,
    },
    /// JSON (de)serialization failure.
    #[error("frame-document JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl FrameDocument {
    /// Validate the header (version + conventions, **before** interpreting
    /// any state), index ranges, uid uniqueness, and finiteness of every
    /// numeric field.
    pub fn validate(&self) -> Result<(), DocError> {
        validate_header(&self.header)?;
        let mut seen = vec![false; self.uids.len()];
        for (i, rec) in self.records.iter().enumerate() {
            validate_record(rec, i, self.uids.len())?;
            let idx = rec.uid_index as usize;
            if seen[idx] {
                return Err(DocError::DuplicateUid {
                    index: rec.uid_index,
                });
            }
            seen[idx] = true;
        }
        Ok(())
    }

    /// Serialize to a JSON string.
    ///
    /// # Panics
    /// Panics if the document fails [`Self::validate`] — most importantly
    /// on any non-finite numeric field. A NaN here is upstream broken
    /// physics; serializing it would launder it into "data".
    pub fn to_json_string(&self) -> String {
        self.validate().unwrap_or_else(|err| {
            panic!(
                "FrameDocument::to_json_string: refusing to serialize an invalid document: {err}"
            )
        });
        serde_json::to_string(self)
            .expect("FrameDocument serialization is infallible after validate()")
    }

    /// Parse from a JSON string and [`Self::validate`] (header conventions
    /// are checked before the caller can touch any state).
    pub fn from_json_str(json: &str) -> Result<Self, DocError> {
        let doc: Self = serde_json::from_str(json)?;
        doc.validate()?;
        Ok(doc)
    }
}

/// Shared header validation (snapshot + series).
pub(crate) fn validate_header(header: &DocHeader) -> Result<(), DocError> {
    if header.schema_version != SCHEMA_VERSION {
        return Err(DocError::UnsupportedVersion {
            found: header.schema_version,
        });
    }
    let expected = Conventions::current();
    let pairs = [
        (
            "translation",
            &header.conventions.translation,
            &expected.translation,
        ),
        ("rotation", &header.conventions.rotation, &expected.rotation),
        (
            "angular_velocity",
            &header.conventions.angular_velocity,
            &expected.angular_velocity,
        ),
        (
            "time_scale",
            &header.conventions.time_scale,
            &expected.time_scale,
        ),
    ];
    for (field, found, want) in pairs {
        if found != want {
            return Err(DocError::ConventionMismatch {
                field,
                found: found.clone(),
                expected: want.clone(),
            });
        }
    }
    finite(header.simtime, || "header.simtime".into())?;
    finite(header.tai_tjt_at_epoch, || "header.tai_tjt_at_epoch".into())?;
    Ok(())
}

/// Shared per-record validation (snapshot + series rows).
pub(crate) fn validate_record(
    rec: &FrameRecord,
    record_pos: usize,
    uid_table_len: usize,
) -> Result<(), DocError> {
    if rec.uid_index as usize >= uid_table_len {
        return Err(DocError::UidIndexOutOfRange {
            record: record_pos,
            index: rec.uid_index,
            len: uid_table_len,
        });
    }
    if let Some(p) = rec.parent {
        if p as usize >= uid_table_len {
            return Err(DocError::ParentIndexOutOfRange {
                record: record_pos,
                index: p,
                len: uid_table_len,
            });
        }
        if p == rec.uid_index {
            return Err(DocError::SelfParent {
                record: record_pos,
                index: rec.uid_index,
            });
        }
    }
    let ctx = |field: &str| format!("record {record_pos} ({}) {field}", rec.name);
    if let Some(e) = rec.epoch {
        finite(e, || ctx("epoch"))?;
    }
    finite3(&rec.trans.position, || ctx("trans.position"))?;
    finite3(&rec.trans.velocity, || ctx("trans.velocity"))?;
    match &rec.rotation {
        CanonicalRotation::Quat(q) => {
            for v in q {
                finite(*v, || ctx("rotation.quat"))?;
            }
        }
        CanonicalRotation::Matrix(m) => {
            for row in m {
                finite3(row, || ctx("rotation.matrix"))?;
            }
        }
    }
    finite3(&rec.ang_vel_this, || ctx("ang_vel_this"))?;
    if let Origin::Integrated {
        attitude_quat,
        ang_vel_body,
    } = &rec.origin
    {
        if let Some(q) = attitude_quat {
            for v in q {
                finite(*v, || ctx("origin.attitude_quat"))?;
            }
        }
        if let Some(w) = ang_vel_body {
            finite3(w, || ctx("origin.ang_vel_body"))?;
        }
    }
    Ok(())
}

fn finite(v: f64, ctx: impl Fn() -> String) -> Result<(), DocError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(DocError::NonFinite(format!("{} = {v}", ctx())))
    }
}

fn finite3(v: &[f64; 3], ctx: impl Fn() -> String) -> Result<(), DocError> {
    for x in v {
        finite(*x, &ctx)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{body_record, pfix_record, record_bits, snapshot};

    #[test]
    fn snapshot_round_trips_bit_exact_both_canonicity_regimes() {
        // The snapshot fixture carries a quaternion-canonical body record
        // AND a matrix-canonical pfix record — both regimes must survive
        // JSON bit-for-bit (RFS-601; shortest-round-trip float printing).
        let doc = snapshot();
        let json = doc.to_json_string();
        let back = FrameDocument::from_json_str(&json).expect("round trip");
        assert_eq!(back.records.len(), doc.records.len());
        for (a, b) in doc.records.iter().zip(&back.records) {
            assert_eq!(record_bits(a), record_bits(b), "record {} drifted", a.name);
        }
        assert_eq!(doc.header.simtime.to_bits(), back.header.simtime.to_bits());
        assert_eq!(
            doc.header.tai_tjt_at_epoch.to_bits(),
            back.header.tai_tjt_at_epoch.to_bits()
        );
        assert_eq!(doc.uids, back.uids);
    }

    #[test]
    fn uid_table_interning_round_trips() {
        // Identity travels by interned table: a type-derived LOCAL uid and
        // external non-LOCAL uids all survive, and records reference the
        // table by index (root's parent is None).
        let doc = snapshot();
        let json = doc.to_json_string();
        let back = FrameDocument::from_json_str(&json).expect("round trip");
        assert_eq!(back.uids, doc.uids);
        assert_eq!(back.records[0].parent, None, "root names no parent");
        assert_eq!(back.records[1].parent, Some(0));
        assert_eq!(back.records[2].parent, Some(0));
        assert!(matches!(
            back.records[1].origin,
            Origin::Derived { ref model } if model == "EarthRNP"
        ));
        assert!(matches!(back.records[0].origin, Origin::Injected));
        assert!(matches!(
            back.records[2].origin,
            Origin::Integrated {
                attitude_quat: Some(_),
                ang_vel_body: Some(_)
            }
        ));
    }

    #[test]
    #[should_panic(expected = "non-finite value")]
    fn non_finite_serialize_panics() {
        let mut doc = snapshot();
        doc.records[2].trans.velocity[1] = f64::NAN;
        let _ = doc.to_json_string();
    }

    #[test]
    fn validate_rejects_out_of_range_uid_index() {
        let mut doc = snapshot();
        doc.records[1].uid_index = 99;
        assert!(matches!(
            doc.validate(),
            Err(DocError::UidIndexOutOfRange {
                record: 1,
                index: 99,
                ..
            })
        ));
    }

    #[test]
    fn validate_rejects_out_of_range_parent() {
        let mut doc = snapshot();
        doc.records[2].parent = Some(99);
        assert!(matches!(
            doc.validate(),
            Err(DocError::ParentIndexOutOfRange {
                record: 2,
                index: 99,
                ..
            })
        ));
    }

    #[test]
    fn validate_rejects_self_parent() {
        let mut doc = snapshot();
        doc.records[1].parent = Some(doc.records[1].uid_index);
        assert!(matches!(doc.validate(), Err(DocError::SelfParent { .. })));
    }

    #[test]
    fn validate_rejects_duplicate_uid() {
        let mut doc = snapshot();
        doc.records[2].uid_index = doc.records[1].uid_index;
        assert!(matches!(doc.validate(), Err(DocError::DuplicateUid { .. })));
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut doc = snapshot();
        doc.header.schema_version = SCHEMA_VERSION + 1;
        let json = serde_json::to_string(&doc).expect("raw serialize");
        assert!(matches!(
            FrameDocument::from_json_str(&json),
            Err(DocError::UnsupportedVersion { found }) if found == SCHEMA_VERSION + 1
        ));
    }

    #[test]
    fn convention_mismatch_rejected_before_state_is_interpreted() {
        let mut doc = snapshot();
        doc.header.conventions.rotation = "scalar-LAST right-transformation".into();
        let json = serde_json::to_string(&doc).expect("raw serialize");
        assert!(matches!(
            FrameDocument::from_json_str(&json),
            Err(DocError::ConventionMismatch {
                field: "rotation",
                ..
            })
        ));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Full-entropy finite f64: any bit pattern that is not NaN/inf,
        /// including subnormals and -0.0.
        fn finite_f64() -> impl Strategy<Value = f64> {
            any::<u64>()
                .prop_map(f64::from_bits)
                .prop_filter("finite", |x| x.is_finite())
        }

        fn arr3() -> impl Strategy<Value = [f64; 3]> {
            [finite_f64(), finite_f64(), finite_f64()]
        }

        fn arr4() -> impl Strategy<Value = [f64; 4]> {
            [finite_f64(), finite_f64(), finite_f64(), finite_f64()]
        }

        proptest! {
            /// RFS-601: every finite f64 bit pattern survives the JSON
            /// round trip bit-for-bit, in both rotation representations
            /// and all origin payloads.
            #[test]
            fn any_finite_f64_round_trips_bit_exact(
                pos in arr3(), vel in arr3(), quat in arr4(),
                m0 in arr3(), m1 in arr3(), m2 in arr3(),
                w in arr3(), wb in arr3(), epoch in finite_f64(),
                simtime in finite_f64(), tjt in finite_f64(),
            ) {
                let mut doc = snapshot();
                doc.header.simtime = simtime;
                doc.header.tai_tjt_at_epoch = tjt;
                doc.records[1] = FrameRecord {
                    rotation: CanonicalRotation::Matrix([m0, m1, m2]),
                    epoch: Some(epoch),
                    ..pfix_record()
                };
                doc.records[2] = FrameRecord {
                    trans: TransRecord { position: pos, velocity: vel },
                    rotation: CanonicalRotation::Quat(quat),
                    ang_vel_this: w,
                    origin: Origin::Integrated {
                        attitude_quat: Some(quat),
                        ang_vel_body: Some(wb),
                    },
                    ..body_record()
                };
                let json = doc.to_json_string();
                let back = FrameDocument::from_json_str(&json).expect("round trip");
                for (a, b) in doc.records.iter().zip(&back.records) {
                    prop_assert_eq!(record_bits(a), record_bits(b));
                }
                prop_assert_eq!(doc.header.simtime.to_bits(), back.header.simtime.to_bits());
                prop_assert_eq!(
                    doc.header.tai_tjt_at_epoch.to_bits(),
                    back.header.tai_tjt_at_epoch.to_bits()
                );
            }
        }
    }
}
