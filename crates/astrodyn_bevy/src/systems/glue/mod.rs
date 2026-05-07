//! Bevy systems that wire ECS plumbing alongside the seven
//! [`AstrodynSet`](crate::AstrodynSet) pipeline stages — frame-tree
//! registration / sync, kinematic-joint drivers and exclusivity
//! guards, despawn observers, and the per-step mass recompute. Each
//! system here is registered at a specific schedule slot by
//! [`crate::AstrodynPlugin`] / [`crate::register_planet_systems`] rather
//! than via `in_set(AstrodynSet::*)`.

pub mod despawn_observers;
pub mod frame_registration;
pub mod joint_kinematics;
pub mod mass_update;

pub use despawn_observers::*;
pub use frame_registration::*;
pub use joint_kinematics::*;
pub use mass_update::*;
