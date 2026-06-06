//! Runtime frame-identity vocabulary derived from the compile-time
//! [`Frame`] tags.
//!
//! [`Frame::DESCRIPTOR`] lifts each sealed frame phantom to a `'static`
//! [`FrameDescriptorStatic`]; [`FrameUid::of`] mints the owned, comparable,
//! hashable identity value the runtime frame layer (issue #659) builds on.
//! This module is purely additive vocabulary — it changes no physics and no
//! existing `RefFrameKind` behavior.
//!
//! ## Identity model
//!
//! A [`FrameUid`] is a *value-level* frame identity: `(namespace, class,
//! role, tag)`. Identities in [`Namespace::LOCAL`] are exclusively
//! type-derived — [`FrameUid::of`] is the only mint for `LOCAL`, and
//! [`FrameUid::external`] refuses it — so a `LOCAL` identity always traces
//! to exactly one concrete sealed frame type. That reservation is what makes
//! identity comparison against [`FrameUid::of`] a sound check: no
//! externally-supplied identity can impersonate a type-derived one.
//!
//! ## Mint policy
//!
//! Not every frame *type* names one physical frame, so each descriptor
//! carries a [`MintPolicy`] telling [`FrameUid::of`] what to do:
//!
//! - [`MintPolicy::Stable`] — the type pins down one physical frame; mint it.
//! - [`MintPolicy::Wildcard`] — the type is a runtime-resolved
//!   storage-boundary wildcard (a frame parameterized by the `SelfRef` /
//!   `SelfPlanet` tags, or `MassNode`); minting would alias distinct
//!   physical frames under one identity, so `of` fails loudly instead.
//! - [`MintPolicy::PerBodyIntegration`] — `IntegrationFrame` only: a real,
//!   kind-distinct frame (RF.10) that resolves per body; publish the
//!   resolved frame's identity instead.
//!
//! JEOD_INV: TS.01 — the mint policy refuses to mint a `FrameUid` for the
//! runtime-resolved-boundary wildcards SelfRef / SelfPlanet / MassNode; see
//! `docs/JEOD_invariants.md` row TS.01 and the lint at
//! `tests/self_ref_self_planet_discipline.rs`.

use core::fmt;

use crate::frame::Frame;

/// Coarse runtime taxonomy of frame *kinds*.
///
/// `#[non_exhaustive]` so a later PR can add a kind (e.g. switch frames)
/// without breaking downstream `match` arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum FrameClass {
    /// The simulation's root inertial node.
    RootInertial,
    /// A planet's inertial (non-rotating, J2000-aligned) frame.
    PlanetInertial,
    /// A barycenter's inertial frame (reserved; no sealed impl yet).
    BarycenterInertial,
    /// A planet-fixed (co-rotating) frame.
    PlanetFixed,
    /// A site-anchored topocentric (ENU) frame.
    Topocentric,
    /// A vehicle body or structural frame.
    Body,
    /// An orbit-relative derived frame (LVLH, NED).
    OrbitRelative,
    /// An externally-supplied frame not derived from a sealed tag.
    External,
}

/// Orthogonal *role* within a [`FrameClass`] — borrowed (`'static`) form
/// used by const descriptors.
///
/// Owned sibling: [`FrameRole`] (see the `From` bridge). The split exists
/// because a `const DESCRIPTOR` must be `Copy`/`'static` while [`FrameUid`]
/// needs owned strings for hashing and (de)serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameRoleStatic {
    /// The canonical/primary frame of its class.
    Primary,
    /// Alternate inertial labeling (JEOD `alt_inertial`; reserved).
    AltInertial,
    /// Alternate planet-fixed labeling (JEOD `alt_pfix`). Also carried by
    /// the legacy `Ecef` marker so its descriptor stays distinct from
    /// `PlanetFixed<Earth>` (identity injectivity).
    AltPfix,
    /// Core (point-mass CoM) body node (JEOD `core_body`; reserved).
    CoreBody,
    /// Composite (CoM) body frame — the integrated frame (JEOD
    /// `composite_body`).
    CompositeBody,
    /// Structural (geometric-origin) body frame (JEOD `structure`).
    Structure,
    /// A discrete point on a vehicle (JEOD vehicle point; reserved).
    VehiclePoint,
    /// Local-vertical / local-horizontal orbit-relative role.
    Lvlh,
    /// North-east-down orbit-relative role.
    Ned,
    /// Caller-named role, borrowed for `'static` descriptors.
    Custom(&'static str),
}

/// Orthogonal *role* within a [`FrameClass`] — owned form carried by
/// [`FrameUid`].
///
/// Borrowed sibling: [`FrameRoleStatic`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum FrameRole {
    /// See [`FrameRoleStatic::Primary`].
    Primary,
    /// See [`FrameRoleStatic::AltInertial`].
    AltInertial,
    /// See [`FrameRoleStatic::AltPfix`].
    AltPfix,
    /// See [`FrameRoleStatic::CoreBody`].
    CoreBody,
    /// See [`FrameRoleStatic::CompositeBody`].
    CompositeBody,
    /// See [`FrameRoleStatic::Structure`].
    Structure,
    /// See [`FrameRoleStatic::VehiclePoint`].
    VehiclePoint,
    /// See [`FrameRoleStatic::Lvlh`].
    Lvlh,
    /// See [`FrameRoleStatic::Ned`].
    Ned,
    /// Caller-named role, owned.
    Custom(Box<str>),
}

impl FrameRole {
    /// `true` iff this owned role equals the borrowed `role`
    /// variant-for-variant (string contents compared for `Custom`).
    /// Zero-allocation sibling of `FrameRole::from(role) == *self`.
    pub fn matches_static(&self, role: FrameRoleStatic) -> bool {
        match (self, role) {
            (FrameRole::Primary, FrameRoleStatic::Primary)
            | (FrameRole::AltInertial, FrameRoleStatic::AltInertial)
            | (FrameRole::AltPfix, FrameRoleStatic::AltPfix)
            | (FrameRole::CoreBody, FrameRoleStatic::CoreBody)
            | (FrameRole::CompositeBody, FrameRoleStatic::CompositeBody)
            | (FrameRole::Structure, FrameRoleStatic::Structure)
            | (FrameRole::VehiclePoint, FrameRoleStatic::VehiclePoint)
            | (FrameRole::Lvlh, FrameRoleStatic::Lvlh)
            | (FrameRole::Ned, FrameRoleStatic::Ned) => true,
            (FrameRole::Custom(owned), FrameRoleStatic::Custom(expected)) => &**owned == expected,
            _ => false,
        }
    }
}

impl From<FrameRoleStatic> for FrameRole {
    fn from(r: FrameRoleStatic) -> Self {
        match r {
            FrameRoleStatic::Primary => FrameRole::Primary,
            FrameRoleStatic::AltInertial => FrameRole::AltInertial,
            FrameRoleStatic::AltPfix => FrameRole::AltPfix,
            FrameRoleStatic::CoreBody => FrameRole::CoreBody,
            FrameRoleStatic::CompositeBody => FrameRole::CompositeBody,
            FrameRoleStatic::Structure => FrameRole::Structure,
            FrameRoleStatic::VehiclePoint => FrameRole::VehiclePoint,
            FrameRoleStatic::Lvlh => FrameRole::Lvlh,
            FrameRoleStatic::Ned => FrameRole::Ned,
            FrameRoleStatic::Custom(s) => FrameRole::Custom(s.into()),
        }
    }
}

/// Identity-space discriminator for [`FrameUid`]s.
///
/// [`Namespace::LOCAL`] (0) is **reserved for type-derived identities**
/// ([`FrameUid::of`]); externally-supplied frames take a non-`LOCAL`
/// namespace allocated by the composing host, so independently-produced
/// identities can never silently alias or impersonate a type-derived one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Namespace(pub u16);

impl Namespace {
    /// Reserved namespace for type-derived identities ([`FrameUid::of`]).
    pub const LOCAL: Namespace = Namespace(0);
}

/// Optional per-instance disambiguator: the planet/vehicle name for
/// parameterized frames, or a producer-chosen key for external frames.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Tag {
    /// No instance tag (singleton frames such as the root inertial node).
    None,
    /// A named instance (e.g. `"Earth"`, `"Iss"`, `"site:KSC-LC39A"`).
    Named(Box<str>),
}

/// Three-way mint behavior of a frame descriptor.
///
/// A boolean cannot express the `IntegrationFrame` case distinctly from the
/// TS.01 wildcards — the two refusals need different diagnostics and
/// different fixes. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MintPolicy {
    /// The type pins down exactly one physical frame; [`FrameUid::of`]
    /// mints its identity.
    Stable,
    /// A runtime-resolved storage-boundary wildcard; [`FrameUid::of`]
    /// refuses (JEOD_INV TS.01).
    Wildcard,
    /// `IntegrationFrame`: resolves per body to `RootInertial` or
    /// `PlanetInertial<P>` (RF.10); [`FrameUid::of`] refuses and directs
    /// the caller to mint the resolved frame's identity.
    PerBodyIntegration,
}

/// The `'static` runtime descriptor attached to every sealed [`Frame`] impl
/// via [`Frame::DESCRIPTOR`].
///
/// For parameterized frames the `tag` is composed from the existing
/// `Planet::NAME` / `Vehicle::NAME` consts and the `mint` from the tag's
/// `IS_WILDCARD` flag, so `define_planet!` / `define_vehicle!` call sites
/// need no changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameDescriptorStatic {
    /// Frame taxonomy.
    pub class: FrameClass,
    /// Orthogonal role within the class.
    pub role: FrameRoleStatic,
    /// Static instance tag (`Some(P::NAME)` / `Some(V::NAME)`), or `None`
    /// for unparameterized frames.
    pub tag: Option<&'static str>,
    /// Mint behavior — what [`FrameUid::of`] does for this type.
    pub mint: MintPolicy,
}

impl FrameDescriptorStatic {
    /// `true` iff this descriptor belongs to a runtime-resolved
    /// storage-boundary wildcard (the TS.01 set). Provided so test and lint
    /// vocabulary stay aligned with the `is_wildcard` language used by the
    /// invariant catalog and issue #659.
    pub const fn is_wildcard(&self) -> bool {
        matches!(self.mint, MintPolicy::Wildcard)
    }
}

/// Maps a parameter tag's wildcard flag to the frame's [`MintPolicy`].
///
/// Used in the generic `impl Frame for …` blocks to compute `DESCRIPTOR`
/// from `P::IS_WILDCARD` / `V::IS_WILDCARD` at compile time.
pub const fn mint_for_param(is_wildcard: bool) -> MintPolicy {
    if is_wildcard {
        MintPolicy::Wildcard
    } else {
        MintPolicy::Stable
    }
}

/// Owned, comparable, hashable runtime frame identity.
///
/// Obtained from a compile-time frame type via [`FrameUid::of`] (always in
/// [`Namespace::LOCAL`]) or supplied by an external producer via
/// [`FrameUid::external`] (never `LOCAL`). Equality and hashing are
/// structural over all four fields, which is sound because the `LOCAL`
/// reservation guarantees type-derived and externally-supplied identities
/// live in disjoint namespaces.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameUid {
    /// Identity-space discriminator ([`Namespace::LOCAL`] iff type-derived).
    pub namespace: Namespace,
    /// Frame taxonomy.
    pub class: FrameClass,
    /// Orthogonal role within the class.
    pub role: FrameRole,
    /// Optional instance tag.
    pub tag: Tag,
}

impl FrameUid {
    /// Mint the runtime identity of the concrete frame type `F`.
    ///
    /// Total for [`MintPolicy::Stable`] frames. Fails loudly otherwise:
    ///
    /// # Panics
    ///
    /// - `F` is a TS.01 storage-boundary wildcard (a frame parameterized by
    ///   the self-referential tags, or the mass-tree scratch tag): the type
    ///   does not name one physical frame, so no identity exists to mint.
    /// - `F` is `IntegrationFrame` (RF.10): the physical frame resolves per
    ///   body; mint the resolved frame's identity instead.
    pub fn of<F: Frame>() -> FrameUid {
        let d = F::DESCRIPTOR;
        match d.mint {
            MintPolicy::Stable => FrameUid {
                namespace: Namespace::LOCAL,
                class: d.class,
                role: d.role.into(),
                tag: match d.tag {
                    Some(name) => Tag::Named(name.into()),
                    None => Tag::None,
                },
            },
            // JEOD_INV: TS.01 — refuses to mint identities for the
            // runtime-resolved storage-boundary wildcard frames.
            MintPolicy::Wildcard => panic!(
                "JEOD_INV TS.01: cannot mint a FrameUid for `{}` (class {:?}): it is a \
                 runtime-resolved storage-boundary wildcard with no concrete identity. \
                 Resolve the entity's real vehicle / planet / per-node frame at the \
                 storage boundary and call `FrameUid::of` on that concrete frame type \
                 instead.",
                F::NAME,
                d.class
            ),
            // JEOD_INV: RF.10 — IntegrationFrame is kind-distinct and
            // resolves per body; its identity is the resolved frame's.
            MintPolicy::PerBodyIntegration => panic!(
                "RF.10: cannot mint a FrameUid for `IntegrationFrame`: it resolves per \
                 body to `RootInertial` or `PlanetInertial<P>`. Publish the resolved \
                 frame's identity — mint `FrameUid::of::<RootInertial>()` or \
                 `FrameUid::of::<PlanetInertial<P>>()` for the body's actual \
                 integration frame instead."
            ),
        }
    }

    /// Construct an externally-supplied identity (a producer-defined frame
    /// outside the sealed compile-time set).
    ///
    /// # Panics
    ///
    /// Panics if handed [`Namespace::LOCAL`], which is reserved for
    /// type-derived identities: allowing external mints in `LOCAL` would
    /// let a producer-supplied frame impersonate a type-derived one and
    /// silently pass identity checks.
    pub fn external(ns: Namespace, class: FrameClass, role: FrameRole, tag: Tag) -> FrameUid {
        assert!(
            ns != Namespace::LOCAL,
            "FrameUid::external was handed Namespace::LOCAL, which is reserved for \
             type-derived identity (FrameUid::of). Pass a non-zero namespace allocated \
             for this producer, or use FrameUid::of::<F>() if the identity comes from \
             a sealed Frame type."
        );
        FrameUid {
            namespace: ns,
            class,
            role,
            tag,
        }
    }

    /// Relabel this identity into another namespace (e.g. re-stamping an
    /// imported tree's identities into the import namespace).
    #[must_use]
    pub fn with_namespace(mut self, ns: Namespace) -> FrameUid {
        self.namespace = ns;
        self
    }

    /// Non-panicking predicate: does this uid equal the identity `F` would
    /// mint?
    ///
    /// Returns `false` — never panics — when `F` is not mintable (a TS.01
    /// wildcard or `IntegrationFrame`): a non-mintable type is never equal
    /// to any concrete identity. Zero-allocation: compares structurally
    /// against `F::DESCRIPTOR` instead of minting a throwaway uid.
    /// Groundwork for the checked typed recovery of issue #661 (PR-2).
    pub fn is<F: Frame>(&self) -> bool {
        let d = F::DESCRIPTOR;
        if !matches!(d.mint, MintPolicy::Stable) {
            return false;
        }
        self.namespace == Namespace::LOCAL
            && self.class == d.class
            && self.role.matches_static(d.role)
            && match (&self.tag, d.tag) {
                (Tag::Named(name), Some(expected)) => &**name == expected,
                (Tag::None, None) => true,
                _ => false,
            }
    }
}

impl fmt::Display for FrameUid {
    /// Dotted display names for diagnostics, aligned with the frame-tree
    /// naming vocabulary used by the `astrodyn` orchestration layer
    /// (`"Earth.inertial"`, `"Earth.pfix"`, `"Iss.composite_body"`).
    ///
    /// The suffix is derived from the **role** — verbatim for every
    /// non-`Primary` role (`Custom` roles print their own string) — with
    /// `Primary` falling back to a class-conventional suffix. The tag's
    /// casing is preserved, and non-`LOCAL` namespaces are prefixed `nsN:`.
    /// Display strings are for humans; identity comparison always uses the
    /// structured fields, never this rendering.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.namespace != Namespace::LOCAL {
            write!(f, "ns{}:", self.namespace.0)?;
        }
        // Role-first: every distinct role keeps a distinct suffix, so a
        // rendered uid never misstates its role (FrameUid::external can
        // construct any class/role combination).
        let suffix: &str = match &self.role {
            FrameRole::Custom(role) => role,
            FrameRole::AltInertial => "inertial.alt",
            FrameRole::AltPfix => "pfix.alt",
            FrameRole::CoreBody => "core_body",
            FrameRole::CompositeBody => "composite_body",
            FrameRole::Structure => "structural",
            FrameRole::VehiclePoint => "point",
            FrameRole::Lvlh => "lvlh",
            FrameRole::Ned => "ned",
            FrameRole::Primary => match self.class {
                FrameClass::RootInertial
                | FrameClass::PlanetInertial
                | FrameClass::BarycenterInertial => "inertial",
                FrameClass::PlanetFixed => "pfix",
                FrameClass::Topocentric => "topo",
                FrameClass::Body => "body",
                FrameClass::OrbitRelative => "orbit_relative",
                FrameClass::External => "external",
            },
        };
        match &self.tag {
            Tag::Named(name) => write!(f, "{name}.{suffix}"),
            Tag::None => write!(f, "{suffix}"),
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    //! JSON round-trips for the owned identity graph, mirroring the
    //! `cartesian_state.rs` serde-test pattern.

    use super::*;
    use crate::frame::{Earth, PlanetInertial, RootInertial};

    #[test]
    fn type_derived_uid_round_trips() {
        let uid = FrameUid::of::<PlanetInertial<Earth>>();
        let json = serde_json::to_string(&uid).expect("FrameUid serializes");
        let back: FrameUid = serde_json::from_str(&json).expect("FrameUid deserializes");
        assert_eq!(uid, back);
    }

    #[test]
    fn external_uid_with_custom_role_round_trips() {
        let uid = FrameUid::external(
            Namespace(7),
            FrameClass::External,
            FrameRole::Custom("sensor_boresight".into()),
            Tag::Named("ext-probe-42".into()),
        );
        let json = serde_json::to_string(&uid).expect("FrameUid serializes");
        let back: FrameUid = serde_json::from_str(&json).expect("FrameUid deserializes");
        assert_eq!(uid, back);
    }

    #[test]
    fn untagged_uid_round_trips() {
        let uid = FrameUid::of::<RootInertial>();
        assert_eq!(uid.tag, Tag::None);
        let json = serde_json::to_string(&uid).expect("FrameUid serializes");
        let back: FrameUid = serde_json::from_str(&json).expect("FrameUid deserializes");
        assert_eq!(uid, back);
    }
}
