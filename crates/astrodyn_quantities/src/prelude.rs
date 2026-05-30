// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` / `<MassNode>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Prelude: `use astrodyn_quantities::prelude::*;` brings in everything mission
//! crates typically need without `PhantomData` or `uom::si::*` noise.

pub use crate::aliases::*;
pub use crate::body_attitude::BodyAttitude;
pub use crate::dims::{GravParam, MassFlowRate, SpecificAngMom, SpecificEnergy};
pub use crate::ext::{Array3Ext, F64Ext, Vec3Ext};
pub use crate::frame::{
    BodyFrame, Earth, Ecef, Frame, IntegrationFrame, Jupiter, Lvlh, Mars, MassNode, Moon, Ned,
    Planet, PlanetFixed, PlanetInertial, RootInertial, Saturn, SelfPlanet, SelfRef,
    StructuralFrame, Sun, Vehicle,
};
pub use crate::frame_transform::FrameTransform;
pub use crate::integ_origin::IntegOrigin;
pub use crate::qty3::Qty3;
pub use crate::quat::{
    JeodQuat, Layout, LeftTransform, NormalizedQuat, Quat, RightTransform, ScalarFirst, ScalarLast,
    Transform,
};
pub use crate::time_scale::{
    SecondsSince, TimeConverter, TimeScale, GMST, GPS, TAI, TDB, TT, UT1, UTC,
};
