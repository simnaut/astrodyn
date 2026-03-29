use bevy::prelude::*;
use jeod_sim::{DragConfig, SrpConfig};

/// Vehicle drag configuration (Cd, area).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct DragConfigC(pub DragConfig);

/// Vehicle SRP configuration (area, Cr).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct SrpConfigC(pub SrpConfig);
