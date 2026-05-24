//! LSODE — Livermore Solver for Ordinary Differential Equations.
//!
//! A port of JEOD's `utils/integration/lsode/` (itself a de-Fortran-ed
//! ODEPACK `DLSODE`): a variable-order, variable-step Nordsieck multistep
//! integrator with an implicit-Adams (non-stiff, orders 1–12) family and a
//! BDF (stiff, orders 1–5) family. Closes issues #200 / #122.
//!
//! Like [`crate::gauss_jackson`], LSODE carries persistent multistep state
//! across steps and runs as a re-entrant state machine: the derivative
//! function is evaluated *outside* the integrator (the pipeline owns
//! gravity), so each point where ODEPACK's Fortran did `CALL F` becomes a
//! return-to-caller that resumes at a recorded entry point.
//!
//! ## Port status (incremental — issue #200)
//!
//! This module is being built leaf-first so each piece is unit-tested in
//! isolation before the integrator core is wired:
//!
//! - [`config`] — `LsodeConfig` + method/corrector enums (JEOD
//!   `LsodeControlDataInterface`). **Done.**
//! - [`coeffs`] — `DCFODE` method/test coefficient generation. **Done**
//!   (unit-tested against known Adams-Moulton coefficients).
//! - `nordsieck` — Nordsieck history array, predictor, interpolation
//!   (`DINTDY`). *Pending.*
//! - `controller` — variable order/step selection. *Pending.*
//! - `corrector` — functional-iteration (Adams) and chord (stiff)
//!   correctors. *Pending.*
//! - `mod` manager — the `DLSODE` driver + re-entry state machine, the
//!   public `LsodeState` and `lsode_translational_step`. *Pending.*
//!
//! Until the integrator core lands, `IntegratorType::Lsode` is not yet a
//! selectable integrator and the `tier3_sim_lsode` placeholder continues to
//! run ABM4. See the plan file's Wave 6 design for the full module map.

pub mod coeffs;
pub mod config;
pub mod error_weights;
pub mod nordsieck;

pub use config::{CorrectorMethod, IntegrationMethod, LsodeConfig};
