//! Bevy `Component` newtypes for the per-body integration configuration
//! that maps a body onto its initial integration frame and the
//! distance-based frame switches that may relocate the body to a
//! different inertial frame mid-trajectory.

use bevy::prelude::*;

#[allow(unused_imports)] // intra-doc-link resolution
use super::{gravity::PlanetFixedRotationC, state::TranslationalStateC};

/// Optional initial integration-frame source for a body (issue #71
/// item 4). Mirrors [`astrodyn::VehicleConfig::integ_source`]: when set
/// to `Some(planet_entity)`, the body integrates in that source's
/// inertial frame; when `None` (or the component is absent), the body
/// integrates in the root inertial frame (the Bevy default).
///
/// Consumed at body-frame registration by `register_body_frames_system`
/// to parent the body's frame entity under the source's frame entity.
/// After registration the live "current integration frame" lookup is
/// the body frame entity's `ChildOf` parent — `gravity_computation_system`
/// and `integration_system` walk that hierarchy via the
/// [`crate::frame_param::FrameOrigin`] SystemParam, and
/// `frame_switch_system` reparents the body's frame entity on switch.
/// `IntegSourceC` is the configuration-time intent only and is
/// intentionally not mutated by the switch.
///
/// **Non-root semantics.** When `Some(...)`, the body's
/// [`TranslationalStateC`] stores position/velocity in the source's
/// inertial frame; the Component's `<PlanetInertial<P>>` phantom
/// encodes the planet-inertial framing structurally, so arithmetic
/// that mixes the body state with a `Position<RootInertial>`
/// gravity-source position no longer compiles without an explicit
/// integration-origin shift (RF.10). Shift sites
/// (`gravity_computation_system`, SRP / solar-beta / earth-lighting)
/// lift the typed origin offset from
/// [`crate::frame_param::FrameOrigin::origin_in_root`] and relabel
/// `<PlanetInertial<P>>` → `<RootInertial>` at the call site.
/// Non-shift consumers (atmosphere, drag, LVLH, geodetic, orbital
/// elements) keep their physics in planet-inertial throughout.
#[derive(Component, Debug, Clone, Default, Deref, DerefMut)]
pub struct IntegSourceC(pub Option<astrodyn::FrameUid>);

/// Distance-based integration-frame switches for a body (issue #71
/// items 3 + Phase C4).
///
/// Each entry triggers a reparent + gravity-controls flip when the body
/// crosses the configured distance. Targets are referenced by the
/// source's inertial-frame `FrameUid` (issue #668) — matching
/// `GravityControlsC`'s identity-keyed semantics. Read by
/// [`crate::frame_switch_system`], which evaluates the predicates
/// against [`crate::frame_param::RelativeFrameState`] and reparents
/// the body's frame entity directly via Bevy `ChildOf`.
#[derive(Component, Debug, Clone, Default, Deref, DerefMut)]
pub struct FrameSwitchesC(pub Vec<astrodyn::FrameSwitchConfig>);
