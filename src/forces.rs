//! Force-collection re-exports.
//!
//! The kernel and typed sibling live in `astrodyn_interactions::forces`.
//! This module preserves the `astrodyn::forces::*` import path used by
//! adapters and mission code.

pub use astrodyn_interactions::forces::{
    collect_and_resolve_forces, collect_and_resolve_forces_typed,
};
