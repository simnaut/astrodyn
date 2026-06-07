#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Frame-document schema: the self-describing serialized form of an
//! astrodyn reference-frame tree.
//!
//! This crate carries **only** the document types and their (de)serialization
//! — no physics. A read-only consumer (renderer, logger, analyzer) links this
//! crate (plus the identity vocabulary in `astrodyn_quantities`) and can
//! interpret a recorded run without the producer's code (spec RFS-602,
//! [Spec-Reference-Frame-Requirements](https://github.com/simnaut/astrodyn/wiki/Spec-Reference-Frame-Requirements)).
//!
//! ## Document model
//!
//! Two forms share one record vocabulary:
//!
//! - [`FrameDocument`] — a **snapshot**: header + interned [`FrameUid`]
//!   table + one [`FrameRecord`] per frame node.
//! - [`FrameSeries`] — **replay v1**: header + uid table + a sequence of
//!   [`FrameSegment`]s, each holding per-epoch record rows. A segment spans
//!   a constant topology; any topology change (a frame-switch reparent, an
//!   attach) closes the segment, so segment boundaries double as seek
//!   keyframes. Streaming topology events are replay v2 (issue #663 scope).
//!
//! **Every state record names the parent uid it is relative to** — snapshot
//! and series alike. Consumers check their folded topology against each
//! record's declared parent, so a missed or misordered topology change is a
//! loud inconsistency, never silent misinterpretation.
//!
//! Per record, state carries its **origin** (RFS-603): `Integrated`
//! (authoritative body store — the tree node is a per-step projection),
//! `Derived` (re-derivable from a model id + the record epoch), or
//! `Injected` (caller-supplied ground truth). Rotation is serialized as
//! [`CanonicalRotation`] — whichever representation was canonical at the
//! write site (the typed path is quaternion-canonical per JEOD_INV RF.04;
//! rotation-model writers like `sync_pfix_rotation` are matrix-canonical) —
//! and the other representation is re-derived on load, which is what makes
//! serialize → reload → continue bit-identical for both regimes (RFS-601).
//!
//! ## Encoding
//!
//! Plain-decimal JSON. Shortest-round-trip float printing makes `f64`
//! round-trips bit-exact (proven by this crate's full-entropy property test
//! and the `cartesian_state_json_round_trips_bit_exact` precedent in
//! `astrodyn_quantities`). Non-finite values are rejected loudly at
//! serialize time — a NaN in a frame document is upstream broken physics,
//! not data. A binary form is a later, additive decision.
//!
//! The header carries the schema version and the numeric conventions
//! **in-band** ([`Conventions`]); [`FrameDocument::validate`] checks them
//! before any state is interpreted, so a consumer that never links astrodyn
//! cannot silently misread a changed convention.
//!
//! ## Per-record / streaming consumers
//!
//! The record types — [`DocHeader`], [`FrameRecord`], [`EpochRow`],
//! [`FrameUid`] and their components — are a **supported, stable surface**
//! for independent serialization, not internals behind the whole-document
//! JSON API. `to_json_string` / `from_json_str` are conveniences over the
//! same `serde` derives; a live feed may serialize records individually
//! (e.g. a binary serde format over a socket) under these rules:
//!
//! - **Handshake = header + uid table.** Records reference identities
//!   positionally (`FrameRecord::uid_index` / `parent` index into the
//!   interned `Vec<FrameUid>`), so transmit [`DocHeader`] and the uid
//!   table once, then per-epoch [`EpochRow`]s. Validate the handshake
//!   *before interpreting any number* via [`validate_header`] and
//!   [`validate_uid_table`], and each arriving row via
//!   [`validate_record`] — the loose-piece equivalents of the
//!   whole-document `validate()` entry points.
//! - **Topology changes are segment boundaries.** A reparent or attach
//!   changes what `parent` means; mirror replay v1's rule on a live feed
//!   by re-sending header + uid table as a fresh keyframe. Per-record
//!   `parent` keeps every row self-checking: verify it against your
//!   folded topology and treat a mismatch as a loud inconsistency, never
//!   a reinterpretation. (Cross-record invariants — cycle-freedom, row
//!   completeness, constant topology within a segment — are the
//!   container `validate()`s' job and become the *consumer's* job on a
//!   stream.)
//! - **Format notes.** Positional formats (postcard/bincode) bind to
//!   field *order*, self-describing formats to field *names*; pin this
//!   crate's version on both ends and gate on
//!   [`DocHeader::schema_version`] at handshake. Binary `f64` encodings
//!   are inherently bit-exact — the shortest-round-trip /
//!   `float_roundtrip` requirement above is JSON-specific. Non-finite
//!   values remain invalid in any encoding.
//! - **Stability.** The wire schema evolves only through
//!   [`SCHEMA_VERSION`]; these Rust types *are* the schema, so field
//!   changes imply a version bump and a semver-visible crate change.
//!   Evolution is additive where possible (enums such as [`Origin`] /
//!   [`CanonicalRotation`] may gain variants behind a version bump);
//!   existing fields are not silently re-shaped.
//!
//! ```
//! use astrodyn_frame_doc::{
//!     validate_header, validate_record, validate_uid_table, CanonicalRotation, Conventions,
//!     DocHeader, EpochRow, FrameRecord, FrameUid, Origin, TransRecord, SCHEMA_VERSION,
//! };
//! use astrodyn_quantities::frame::RootInertial;
//!
//! // ── Producer side: handshake, then rows ──
//! let header = DocHeader {
//!     schema_version: SCHEMA_VERSION,
//!     conventions: Conventions::current(),
//!     simtime: 0.0,
//!     tai_tjt_at_epoch: 11544.499257592593,
//! };
//! let uids = vec![FrameUid::of::<RootInertial>()];
//! let row = EpochRow {
//!     simtime: 0.0,
//!     records: vec![FrameRecord {
//!         name: "root".into(),
//!         uid_index: 0,
//!         parent: None,
//!         epoch: Some(0.0),
//!         trans: TransRecord {
//!             position: [0.0; 3],
//!             velocity: [0.0; 3],
//!         },
//!         rotation: CanonicalRotation::Quat([1.0, 0.0, 0.0, 0.0]),
//!         ang_vel_this: [0.0; 3],
//!         origin: Origin::Injected,
//!     }],
//! };
//!
//! // ── Consumer side: validate the handshake before any number, then
//! //    each arriving row against the handshake's uid table. ──
//! validate_header(&header).expect("handshake header");
//! validate_uid_table(&uids).expect("handshake uid table");
//! for (pos, rec) in row.records.iter().enumerate() {
//!     validate_record(rec, pos, uids.len()).expect("arriving row");
//!     // ...then check rec.parent against your folded topology.
//! }
//! ```

mod document;
mod series;

pub use document::{
    validate_header, validate_record, validate_uid_table, CanonicalRotation, Conventions, DocError,
    DocHeader, FrameDocument, FrameRecord, Origin, TransRecord, SCHEMA_VERSION,
};
pub use series::{EpochRow, FrameSegment, FrameSeries, SeriesBuilder};

// Re-export the identity vocabulary the wire types embed, so a document
// consumer needs only this crate in its dependency list.
pub use astrodyn_quantities::frame_descriptor::{FrameClass, FrameRole, FrameUid, Namespace, Tag};

#[cfg(test)]
pub(crate) mod test_fixtures {
    //! Shared builders for the in-crate unit tests.

    use crate::*;

    /// Three-uid table: a type-derived root, an external planet-fixed
    /// frame, and an external body frame.
    pub fn uid_table() -> Vec<FrameUid> {
        use astrodyn_quantities::frame::RootInertial;
        vec![
            FrameUid::of::<RootInertial>(),
            FrameUid::external(
                Namespace(2),
                FrameClass::PlanetFixed,
                FrameRole::Primary,
                Tag::Named("Earth".into()),
            ),
            FrameUid::external(
                Namespace(2),
                FrameClass::Body,
                FrameRole::CompositeBody,
                Tag::Named("iss".into()),
            ),
        ]
    }

    pub fn header() -> DocHeader {
        DocHeader {
            schema_version: SCHEMA_VERSION,
            conventions: Conventions::current(),
            simtime: 1_234.5,
            tai_tjt_at_epoch: 213.818,
        }
    }

    pub fn root_record() -> FrameRecord {
        FrameRecord {
            name: "root".into(),
            uid_index: 0,
            parent: None,
            epoch: Some(1_234.5),
            trans: TransRecord {
                position: [0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
            },
            rotation: CanonicalRotation::Quat([1.0, 0.0, 0.0, 0.0]),
            ang_vel_this: [0.0, 0.0, 0.0],
            origin: Origin::Injected,
        }
    }

    /// Matrix-canonical pfix record with non-trivial values.
    pub fn pfix_record() -> FrameRecord {
        FrameRecord {
            name: "Earth.pfix".into(),
            uid_index: 1,
            parent: Some(0),
            epoch: Some(1_234.5),
            trans: TransRecord {
                position: [0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
            },
            rotation: CanonicalRotation::Matrix([
                [0.527_472_1, 0.849_573_2, 1.302_4e-3],
                [-0.849_572_9, 0.527_473_3, -8.117_7e-4],
                [-1.376_55e-3, -6.782_1e-4, 0.999_998_8],
            ]),
            ang_vel_this: [0.0, 0.0, 7.292_115_1e-5],
            origin: Origin::Derived {
                model: "EarthRNP".into(),
            },
        }
    }

    /// Quaternion-canonical 6-DOF body record with non-trivial values.
    pub fn body_record() -> FrameRecord {
        FrameRecord {
            name: "body_0.integ".into(),
            uid_index: 2,
            parent: Some(0),
            epoch: Some(1_234.5),
            trans: TransRecord {
                position: [6.778_137_4e6, -42.25, 1_234_567.875],
                velocity: [-0.5, 7_546.125, 3.0],
            },
            rotation: CanonicalRotation::Quat([
                0.879_447_045_2,
                0.279_848_133_6,
                0.364_705_199_4,
                0.115_916_895_9,
            ]),
            ang_vel_this: [1.1e-3, -2.2e-3, 3.3e-3],
            origin: Origin::Integrated {
                attitude_quat: Some([
                    0.879_447_045_2,
                    0.279_848_133_6,
                    0.364_705_199_4,
                    0.115_916_895_9,
                ]),
                ang_vel_body: Some([1.1e-3, -2.2e-3, 3.3e-3]),
            },
        }
    }

    pub fn snapshot() -> FrameDocument {
        FrameDocument {
            header: header(),
            uids: uid_table(),
            records: vec![root_record(), pfix_record(), body_record()],
        }
    }

    /// Every f64 in a record, as bits, for bit-exact comparisons
    /// (PartialEq would accept `-0.0 == 0.0`; bits don't lie).
    pub fn record_bits(rec: &FrameRecord) -> Vec<u64> {
        let mut bits = Vec::new();
        let push3 = |v: &[f64; 3], out: &mut Vec<u64>| {
            out.extend(v.iter().map(|x| x.to_bits()));
        };
        if let Some(e) = rec.epoch {
            bits.push(e.to_bits());
        }
        push3(&rec.trans.position, &mut bits);
        push3(&rec.trans.velocity, &mut bits);
        match &rec.rotation {
            CanonicalRotation::Quat(q) => bits.extend(q.iter().map(|x| x.to_bits())),
            CanonicalRotation::Matrix(m) => {
                for row in m {
                    bits.extend(row.iter().map(|x| x.to_bits()));
                }
            }
        }
        push3(&rec.ang_vel_this, &mut bits);
        if let Origin::Integrated {
            attitude_quat,
            ang_vel_body,
        } = &rec.origin
        {
            if let Some(q) = attitude_quat {
                bits.extend(q.iter().map(|x| x.to_bits()));
            }
            if let Some(w) = ang_vel_body {
                bits.extend(w.iter().map(|x| x.to_bits()));
            }
        }
        bits
    }
}
