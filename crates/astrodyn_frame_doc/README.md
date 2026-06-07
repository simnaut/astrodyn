# astrodyn_frame_doc

Frame-document schema for the
[astrodyn](https://github.com/simnaut/astrodyn) orbital-dynamics
workspace: the self-describing serialized form of a reference-frame
tree — snapshots and replay series — carrying **identity, topology,
origin, and epoch on every record**.

This crate holds **only the document types and their
(de)serialization** — no physics. A read-only consumer (renderer,
logger, analyzer, live telemetry view) links this crate plus the
identity vocabulary it re-exports from `astrodyn_quantities`, and can
interpret a recorded run without ever depending on the producer's
code.

## Document model

Two forms share one record vocabulary:

- **`FrameDocument`** — a snapshot: header + interned `FrameUid` table
  + one `FrameRecord` per frame node.
- **`FrameSeries`** — replay v1: header + uid table + a sequence of
  topology-stable `FrameSegment`s of per-epoch rows. Any topology
  change (a frame-switch reparent, an attach) closes the segment, so
  segment boundaries double as seek keyframes.

Every record names the parent uid its state is relative to, carries
its epoch, and declares its **origin**: `Integrated` (authoritative
body state), `Derived` (re-derivable from a model id + epoch), or
`Injected` (caller-supplied ground truth). Rotation serializes as
whichever representation was canonical at the write site
(`CanonicalRotation::{Quat, Matrix}`) and the other is re-derived on
load — what makes serialize → reload → continue **bit-identical**.

## Per-record / streaming consumers

The record types (`DocHeader`, `FrameRecord`, `EpochRow`, `FrameUid`,
…) are a **supported, stable surface** for independent serialization —
e.g. a live binary feed that sends the header + uid table as a
handshake and then streams per-epoch rows. The piecewise validators
(`validate_header`, `validate_uid_table`, `validate_record`) are the
loose-piece equivalents of the whole-document `validate()` entry
points. See the crate docs for the handshake, keyframe, format, and
stability rules.

## Encoding and stability

Whole-document convenience API is plain-decimal JSON; `f64` round-trips
are bit-exact (consumers parsing JSON themselves need `serde_json`'s
`float_roundtrip` feature). Binary `f64` encodings are inherently
bit-exact. Non-finite values are rejected loudly in any encoding. The
wire schema evolves only through `SCHEMA_VERSION`; these types *are*
the schema, so changes are semver-visible on the crate and additive
where possible.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
