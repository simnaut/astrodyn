use bevy::prelude::*;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum JeodSet {
    Environment,
    ForceCollection,
    Integration,
    DerivedState,
}
