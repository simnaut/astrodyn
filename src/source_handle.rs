//! Typed wrapper around source-table indices used by [`crate::VehicleBuilder`].
//!
//! `VehicleBuilder` methods that reference a gravity source —
//! [`crate::VehicleBuilder::integ_source`],
//! [`crate::VehicleBuilder::frame_switches`],
//! [`crate::VehicleBuilder::geodetic`],
//! [`crate::VehicleBuilder::orbital_elements`],
//! [`crate::VehicleBuilder::shadow`], plus `GravityControl::new_spherical`
//! / `new_nonspherical` / `new_third_body` — historically took a bare
//! `usize`. The bare `usize` left two failure modes silent at the type
//! level: passing the wrong index against the `&[Entity]` slice handed
//! to the eventual `spawn_bevy(..., &[earth, moon, sun])`, and (later
//! on, with multi-planet scenarios) passing a Moon-tagged index where
//! an Earth-tagged one was required.
//!
//! [`SourceHandle`] is the structural fix: a newtype that names the
//! intent ("an index into the per-config source table") at the type
//! level. Callers reach the underlying `usize` only through
//! [`SourceHandle::central`] or [`SourceHandle::index`] (or the
//! `From<usize>` blanket); there is no `Deref<Target = usize>` and no
//! `pub` field, so a stray `5` cannot drift into a method that wants a
//! source identifier. Existing callers continue to compile via
//! `From<usize>`, so the migration is mechanical and reversible.
//!
//! Planet-generic typing — `SourceHandle<P: Planet>` so the compiler
//! refuses to pass a Moon handle to `spawn_bevy::<Earth>` — is a
//! future evolution; this crate currently exposes only the untyped
//! form.

/// Typed wrapper around a per-config gravity-source-table index.
///
/// Constructed via [`SourceHandle::central`] (index 0, the central
/// body of the per-config slice) or [`SourceHandle::index`]. The
/// `From<usize>` blanket is provided so existing builder callsites
/// that pass a bare `usize` continue to compile during incremental
/// migration; new mission code should prefer
/// [`SourceHandle::central`] / [`SourceHandle::index`] for
/// self-documenting intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceHandle(usize);

impl SourceHandle {
    /// The central body of the per-config gravity-source slice — always
    /// index `0`. Most single-planet missions only need this constructor.
    pub const fn central() -> Self {
        Self(0)
    }

    /// A specific index into the per-config gravity-source slice.
    ///
    /// The slice is the `&[Entity]` (Bevy) or registration order
    /// (runner) handed to the consumer at materialization time;
    /// out-of-range values are caught at that boundary, not here.
    pub const fn index(n: usize) -> Self {
        Self(n)
    }

    /// Drop the wrapper and return the underlying `usize`.
    ///
    /// `pub(crate)` because only the gateway should be reaching back
    /// into the `usize` representation — mission code stays on the
    /// typed surface.
    pub(crate) const fn into_raw(self) -> usize {
        self.0
    }
}

impl From<usize> for SourceHandle {
    fn from(n: usize) -> Self {
        Self(n)
    }
}

/// Lower a [`SourceHandle`] back to the underlying `usize` source-table
/// index. Used by builder methods and `GravityControl` constructors
/// that ingest `impl Into<SourceId>` and need the typed handle to
/// participate in the same conversion path as bare-`usize` callers.
impl From<SourceHandle> for usize {
    fn from(h: SourceHandle) -> Self {
        h.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `central()` is the index-0 constructor — the most common shape
    /// for single-planet missions.
    #[test]
    fn central_is_zero() {
        assert_eq!(SourceHandle::central().into_raw(), 0);
    }

    /// `index(n)` round-trips through `into_raw()`.
    #[test]
    fn index_round_trips() {
        for n in [0_usize, 1, 2, 7, 42] {
            assert_eq!(SourceHandle::index(n).into_raw(), n);
        }
    }

    /// `From<usize>` is the backward-compatibility shim — any bare
    /// `usize` callsite must keep compiling and produce the same
    /// underlying index as `SourceHandle::index`.
    #[test]
    fn from_usize_preserves_index() {
        for n in [0_usize, 1, 5, 100] {
            let via_from: SourceHandle = n.into();
            let via_index = SourceHandle::index(n);
            assert_eq!(via_from, via_index);
            assert_eq!(via_from.into_raw(), n);
        }
    }
}
