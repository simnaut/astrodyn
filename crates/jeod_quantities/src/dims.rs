//! Custom quantity dimensions not already present in `uom::si`.
//!
//! Dimensions are compile-time 7-tuples of `typenum` integers indexing
//! `ISQ<L, M, T, I, Th, N, J>`. Each alias below names a physical quantity
//! used by orbital mechanics that `uom::si` does not predefine.
//!
//! The `*Dim` aliases can be used as the dimension parameter of `Qty3<D, F>`;
//! the companion scalar aliases (`GravParam`, `SpecificAngMom`, …) model the
//! corresponding `Quantity<_, SI<f64>, f64>` values used in scalar math.

use typenum::{N1, N2, P1, P2, P3, Z0};
use uom::si::{Quantity, ISQ, SI};

/// Gravitational parameter μ = GM (L³T⁻²). Base SI unit: m³/s².
pub type GravParamDim = ISQ<P3, Z0, N2, Z0, Z0, Z0, Z0>;

/// Specific angular momentum h = |r × v| / m (L²T⁻¹). Base SI unit: m²/s.
pub type SpecificAngMomDim = ISQ<P2, Z0, N1, Z0, Z0, Z0, Z0>;

/// Specific energy ε (L²T⁻²). Base SI unit: J/kg = m²/s².
pub type SpecificEnergyDim = ISQ<P2, Z0, N2, Z0, Z0, Z0, Z0>;

/// Mass flow rate ṁ (MT⁻¹). Base SI unit: kg/s.
pub type MassFlowRateDim = ISQ<Z0, P1, N1, Z0, Z0, Z0, Z0>;

// --- Scalar `Quantity` aliases -----------------------------------------------

/// Scalar gravitational parameter (e.g. `μ_Earth ≈ 3.986e14 m³/s²`).
pub type GravParam = Quantity<GravParamDim, SI<f64>, f64>;

/// Scalar specific angular momentum.
pub type SpecificAngMom = Quantity<SpecificAngMomDim, SI<f64>, f64>;

/// Scalar specific energy.
pub type SpecificEnergy = Quantity<SpecificEnergyDim, SI<f64>, f64>;

/// Scalar mass-flow rate.
pub type MassFlowRate = Quantity<MassFlowRateDim, SI<f64>, f64>;
