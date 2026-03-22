use glam::{DMat3, DVec3};
use jeod_math::JeodQuat;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefFrameTrans {
    pub position: DVec3, // m, in parent frame
    pub velocity: DVec3, // m/s, in parent frame
}

impl Default for RefFrameTrans {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefFrameRot {
    pub q_parent_this: JeodQuat, // left transformation quaternion
    pub t_parent_this: DMat3,    // transformation matrix
    pub ang_vel_this: DVec3,     // rad/s, in this frame
}

impl Default for RefFrameRot {
    fn default() -> Self {
        Self {
            q_parent_this: JeodQuat::identity(),
            t_parent_this: DMat3::IDENTITY,
            ang_vel_this: DVec3::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RefFrameState {
    pub trans: RefFrameTrans,
    pub rot: RefFrameRot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefFrameInfo {
    pub name: String,
    pub kind: RefFrameKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefFrameKind {
    Inertial,
    PlanetFixed,
    Body,
}
