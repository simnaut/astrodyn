//! Sealed-trait pattern: downstream crates cannot implement our marker traits
//! because they cannot name `Sealed`. This closes the frame/time-scale/quaternion
//! marker-trait sets against unintended extensions.

pub trait Sealed {}
