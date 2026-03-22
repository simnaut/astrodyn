use bevy::prelude::*;
use jeod_frames::RefFrameState;

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct RefFrameStateC(pub RefFrameState);

#[derive(Component, Debug, Clone)]
pub struct RefFrameNameC(pub String);
