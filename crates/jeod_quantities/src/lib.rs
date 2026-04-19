//! # jeod_quantities
//!
//! Dimensional-analysis and phantom-tag foundation for `bevy_jeod`.
//!
//! This crate sits at the bottom of the workspace dependency graph. It provides:
//!
//! - Reference-frame and time-scale phantom markers (`Inertial`, `Ecef`, `TAI`, `TT`, ...)
//! - `uom`-backed componentwise 3-vectors `Qty3<D, F>` (`Position<F>`, `Velocity<F>`, ...)
//! - Quaternion convention tags (`ScalarFirst`/`ScalarLast`, `LeftTransform`/`RightTransform`)
//!   plus a `NormalizedQuat` constructor-gated witness
//! - Typed `FrameTransform<From, To>` composing only when inner frames match
//! - The `F64Ext` facade (`400.0.km()`, `51.6.deg()`, `420_000.0.kg()`)
//! - Compiler error messages in physics language via `#[diagnostic::on_unimplemented]`
//!
//! No crate in the workspace consumes these types yet — Phase 0 of the
//! type-system refactor (GitHub issues #101 / #102) is purely additive.

#![forbid(unsafe_code)]

mod sealed;

pub mod aliases;
pub mod diagnostics;
pub mod dims;
pub mod ext;
pub mod frame;
pub mod frame_transform;
pub mod ops;
pub mod prelude;
pub mod qty3;
pub mod quat;
pub mod time_scale;

pub use aliases::*;
pub use dims::*;
pub use frame::*;
pub use frame_transform::*;
pub use qty3::*;
pub use quat::*;
pub use time_scale::*;
