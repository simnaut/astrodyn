//! Generic surface geometry model.
//!
//! Port of JEOD's surface model hierarchy (`models/utils/surface_model/`) which
//! provides shape primitives (`FlatPlate`, `FlatPlateCircular`, `Cylinder`)
//! used by the aerodynamics, radiation-pressure, and contact interaction
//! models. JEOD inherits shapes from a base `Facet` class (position + area +
//! temperature) and interaction models attach their own parameters per facet.
//!
//! This module provides a pure-geometry layer: shape definition, position,
//! projected-area math, and optional articulation. Force/torque computation
//! remains in `flat_plate_aero.rs` and `radiation_pressure.rs` — the intent
//! is that a future per-shape aero or SRP routine can query
//! [`SurfaceShape::projected_area`] and [`SurfaceShape::effective_normal`]
//! instead of duplicating geometry logic.
//!
//! JEOD's own surface model contains only `Cylinder` as a non-plate shape
//! (no cone type). The [`SurfaceShape::ConicalFrustum`] variant here is an
//! extension to support truncated-cone geometries commonly used in launch
//! vehicle SRP/drag decks.

use glam::{DQuat, DVec3};

/// Shape of a single surface element in the vehicle structural frame.
///
/// All dimensions are in meters. Unit vectors (`normal`, `axis`) are assumed
/// to be pre-normalized by the caller; the projected-area math treats them as
/// direction vectors and re-normalizes flow directions locally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurfaceShape {
    /// A flat plate of the given area, with an outward-facing unit normal.
    FlatPlate {
        /// Plate area (m²).
        area: f64,
        /// Outward-facing unit normal in the structural frame.
        normal: DVec3,
    },
    /// A solid cylinder with a circular cross section.
    ///
    /// The cylinder is open in both directions along `axis`; end caps are
    /// treated as circular disks of area `pi * radius²` attached at each end.
    Cylinder {
        /// Cylinder radius (m).
        radius: f64,
        /// Cylinder length along `axis` (m).
        length: f64,
        /// Cylinder axis as a unit vector in the structural frame.
        axis: DVec3,
    },
    /// A truncated cone (conical frustum).
    ///
    /// `radius_bottom` is the radius at the near-axis end, `radius_top` at
    /// the far-axis end; `length` is the distance along `axis` between the
    /// two end caps. Setting `radius_top == radius_bottom` recovers a
    /// cylinder; setting `radius_top == 0` recovers a full cone.
    ConicalFrustum {
        /// Radius at the `-axis` end cap (m).
        radius_bottom: f64,
        /// Radius at the `+axis` end cap (m).
        radius_top: f64,
        /// Length along `axis` between the two caps (m).
        length: f64,
        /// Cone axis as a unit vector in the structural frame, pointing from
        /// bottom cap to top cap.
        axis: DVec3,
    },
}

/// A shape plus its placement in the vehicle structural frame.
///
/// For a flat plate the `center` is the center of pressure; for a cylinder
/// or frustum it is the geometric centroid along the axis (midway between
/// the two end caps).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceFacet {
    /// Geometric shape and its orientation.
    pub shape: SurfaceShape,
    /// Center position in the structural frame (m). For flat plates this is
    /// the center of pressure; for cylinders/frusta it is the midpoint of
    /// the axis.
    pub center: DVec3,
}

/// Current articulation state of a facet.
///
/// A simple single-axis revolute joint model: `angle` is the current rotation
/// (rad) about `axis` (unit vector in the structural frame), and `rate` is
/// the angular rate (rad/s) for propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArticulationState {
    /// Rotation axis (unit vector, structural frame).
    pub axis: DVec3,
    /// Current rotation angle about `axis` (rad).
    pub angle: f64,
    /// Angular rate about `axis` (rad/s). Used by [`ArticulatedFacet::advance`].
    pub rate: f64,
}

impl ArticulationState {
    /// Quaternion representing the current articulation rotation.
    #[inline]
    pub fn rotation(&self) -> DQuat {
        // Guard against a zero-length axis: treat as identity.
        let len2 = self.axis.length_squared();
        if len2 <= 0.0 {
            DQuat::IDENTITY
        } else {
            DQuat::from_axis_angle(self.axis / len2.sqrt(), self.angle)
        }
    }
}

/// A surface facet with optional revolute articulation.
///
/// When `articulation` is `Some`, the stored `facet` represents the *base*
/// pose; the current pose is obtained by applying the articulation rotation
/// to the shape's orientation vector (`normal` for a plate, `axis` for a
/// cylinder/frustum). The facet's `center` position is **not** rotated — a
/// moving hinge point must be modeled by the caller.
///
/// This mirrors JEOD `FlatPlate::update_articulation_internal()`, which
/// rotates the local normal into the vehicle structural frame via the
/// articulated mass-body orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArticulatedFacet {
    /// Base facet (pre-articulation).
    pub facet: SurfaceFacet,
    /// Optional articulation state.
    pub articulation: Option<ArticulationState>,
}

impl ArticulatedFacet {
    /// Construct a fixed (non-articulated) facet.
    #[inline]
    pub fn fixed(facet: SurfaceFacet) -> Self {
        Self {
            facet,
            articulation: None,
        }
    }

    /// Construct an articulated facet with the given initial state.
    #[inline]
    pub fn articulated(facet: SurfaceFacet, articulation: ArticulationState) -> Self {
        Self {
            facet,
            articulation: Some(articulation),
        }
    }

    /// Current shape with articulation applied to its orientation vector.
    ///
    /// For a flat plate the `normal` is rotated; for a cylinder or frustum
    /// the `axis` is rotated. The shape's dimensional parameters (`area`,
    /// `radius`, `length`) are unaffected.
    pub fn current_shape(&self) -> SurfaceShape {
        match self.articulation {
            None => self.facet.shape,
            Some(art) => {
                let q = art.rotation();
                match self.facet.shape {
                    SurfaceShape::FlatPlate { area, normal } => SurfaceShape::FlatPlate {
                        area,
                        normal: q * normal,
                    },
                    SurfaceShape::Cylinder {
                        radius,
                        length,
                        axis,
                    } => SurfaceShape::Cylinder {
                        radius,
                        length,
                        axis: q * axis,
                    },
                    SurfaceShape::ConicalFrustum {
                        radius_bottom,
                        radius_top,
                        length,
                        axis,
                    } => SurfaceShape::ConicalFrustum {
                        radius_bottom,
                        radius_top,
                        length,
                        axis: q * axis,
                    },
                }
            }
        }
    }

    /// Advance the articulation angle by `dt * rate`. No-op if not articulated.
    pub fn advance(&mut self, dt: f64) {
        if let Some(ref mut art) = self.articulation {
            art.angle += art.rate * dt;
        }
    }
}

impl SurfaceShape {
    /// Project the shape's frontal area onto the plane perpendicular to
    /// `flow_direction`.
    ///
    /// `flow_direction` may be any non-zero vector; it is normalized inside.
    /// The result is always non-negative.
    ///
    /// Formulas:
    ///
    /// * **Flat plate** — `area * |n · v̂|`, where `n` is the plate normal
    ///   and `v̂` is the flow-direction unit vector. A plate is equally
    ///   visible from either side.
    /// * **Cylinder** — the silhouette of a finite cylinder is the union
    ///   of a rectangular side projection and a circular end-cap projection.
    ///   With `θ` the angle between the flow direction and the cylinder
    ///   axis,
    ///   ```text
    ///   A_proj = 2·r·L·|sin θ| + π·r²·|cos θ|
    ///   ```
    ///   The end-cap term represents the forward-facing circular cap; the
    ///   far cap is occluded by the body for an opaque convex shape.
    /// * **Conical frustum** — end caps are circles of radii `r_b` and
    ///   `r_t`; only the forward-facing cap contributes the `π·r²·|cos θ|`
    ///   term (chosen by the sign of `axis · v̂`). The lateral surface
    ///   projects to a trapezoid of average width `(r_b + r_t)` and length
    ///   `L·|sin θ|`:
    ///   ```text
    ///   A_proj = (r_b + r_t)·L·|sin θ| + π·r_forward²·|cos θ|
    ///   ```
    ///   where `r_forward = r_t` when `axis · v̂ > 0` (flow hits the top
    ///   cap end-on from behind, meaning the top cap is the rear and the
    ///   bottom cap is forward) and `r_forward = r_b` otherwise. Because
    ///   `v̂` points *in* the flow direction, the forward-facing cap is the
    ///   one whose outward normal has negative dot with `v̂` — i.e.
    ///   `r_forward = r_t` when `axis · v̂ < 0`, else `r_b`.
    pub fn projected_area(&self, flow_direction: DVec3) -> f64 {
        let flow_len2 = flow_direction.length_squared();
        if flow_len2 <= 0.0 {
            return 0.0;
        }
        let v_hat = flow_direction / flow_len2.sqrt();

        match *self {
            SurfaceShape::FlatPlate { area, normal } => {
                let n = normalize_or_zero(normal);
                area * n.dot(v_hat).abs()
            }
            SurfaceShape::Cylinder {
                radius,
                length,
                axis,
            } => {
                let a = normalize_or_zero(axis);
                let cos_theta = a.dot(v_hat);
                // |sin θ| = sqrt(max(0, 1 - cos²θ))  (clamp for FP error)
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                2.0 * radius * length * sin_theta
                    + std::f64::consts::PI * radius * radius * cos_theta.abs()
            }
            SurfaceShape::ConicalFrustum {
                radius_bottom,
                radius_top,
                length,
                axis,
            } => {
                let a = normalize_or_zero(axis);
                let cos_theta = a.dot(v_hat);
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                // Forward-facing cap: outward normal has negative dot with flow.
                // Bottom cap normal = -axis  → dot = -cos_theta
                // Top cap normal    = +axis  → dot = +cos_theta
                // The cap whose normal opposes the flow is the one the flow
                // strikes head-on. When cos_theta > 0, the bottom cap's normal
                // opposes the flow → bottom cap is forward.
                let r_forward = if cos_theta > 0.0 {
                    radius_bottom
                } else {
                    radius_top
                };
                (radius_bottom + radius_top) * length * sin_theta
                    + std::f64::consts::PI * r_forward * r_forward * cos_theta.abs()
            }
        }
    }

    /// Effective outward-facing unit normal for applying a projected force.
    ///
    /// For a flat plate this is simply the plate normal (flipped if it
    /// points away from the flow so that `n · (-v̂) ≥ 0`). For a cylinder
    /// or frustum the "effective normal" is the direction anti-parallel to
    /// the flow (i.e. the unit vector along which a pressure-area scalar
    /// should be multiplied to give a force opposing the flow).
    ///
    /// Returns `DVec3::ZERO` when the flow direction is zero-length or the
    /// shape's orientation vector is zero-length.
    pub fn effective_normal(&self, flow_direction: DVec3) -> DVec3 {
        let flow_len2 = flow_direction.length_squared();
        if flow_len2 <= 0.0 {
            return DVec3::ZERO;
        }
        let v_hat = flow_direction / flow_len2.sqrt();

        match *self {
            SurfaceShape::FlatPlate { normal, .. } => {
                let n = normalize_or_zero(normal);
                // Ensure the effective normal points against the flow
                // (i.e. "into the wind") so that force = -p·A·n_eff opposes motion.
                if n.dot(v_hat) > 0.0 {
                    -n
                } else {
                    n
                }
            }
            SurfaceShape::Cylinder { .. } | SurfaceShape::ConicalFrustum { .. } => {
                // For a body of revolution the net pressure force is along
                // the flow direction by symmetry, so the effective normal
                // is simply -v̂.
                -v_hat
            }
        }
    }
}

/// Normalize a vector, returning [`DVec3::ZERO`] if it has zero length.
#[inline]
fn normalize_or_zero(v: DVec3) -> DVec3 {
    let len2 = v.length_squared();
    if len2 <= 0.0 {
        DVec3::ZERO
    } else {
        v / len2.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_4, PI};

    const TOL: f64 = 1e-12;

    // ── Flat plate ────────────────────────────────────────────────────

    /// Flat plate with normal along flow: projected area = full area.
    #[test]
    fn flat_plate_projected_area_normal_flow() {
        let shape = SurfaceShape::FlatPlate {
            area: 10.0,
            normal: DVec3::X,
        };
        let projected = shape.projected_area(DVec3::X);
        assert!(
            (projected - 10.0).abs() < TOL,
            "Expected 10.0, got {projected}"
        );
    }

    /// Flat plate with normal perpendicular to flow: projected area = 0.
    #[test]
    fn flat_plate_projected_area_parallel_flow() {
        let shape = SurfaceShape::FlatPlate {
            area: 10.0,
            normal: DVec3::X,
        };
        let projected = shape.projected_area(DVec3::Y);
        assert!(projected.abs() < TOL, "Expected 0, got {projected}");
    }

    /// Plate is equally visible from either side: flipping normal gives
    /// the same projected area.
    #[test]
    fn flat_plate_projected_area_symmetric() {
        let shape = SurfaceShape::FlatPlate {
            area: 7.5,
            normal: DVec3::new(1.0, 1.0, 0.0).normalize(),
        };
        let v = DVec3::X;
        let a_forward = shape.projected_area(v);
        let a_backward = shape.projected_area(-v);
        assert!((a_forward - a_backward).abs() < TOL);
    }

    /// Flat plate at 45 degrees: projected area = area * cos(45°).
    #[test]
    fn flat_plate_projected_area_45_degrees() {
        let shape = SurfaceShape::FlatPlate {
            area: 10.0,
            normal: DVec3::new(1.0, 1.0, 0.0).normalize(),
        };
        let projected = shape.projected_area(DVec3::X);
        let expected = 10.0 * (1.0 / 2.0_f64.sqrt());
        assert!(
            (projected - expected).abs() < TOL,
            "Expected {expected}, got {projected}"
        );
    }

    // ── Cylinder ──────────────────────────────────────────────────────

    /// Cylinder with axis along flow: projected area = pi*r^2 (end cap only).
    #[test]
    fn cylinder_projected_area_axial_flow() {
        let r = 2.0;
        let l = 5.0;
        let shape = SurfaceShape::Cylinder {
            radius: r,
            length: l,
            axis: DVec3::X,
        };
        let projected = shape.projected_area(DVec3::X);
        let expected = PI * r * r;
        assert!(
            (projected - expected).abs() < TOL,
            "Axial flow: expected {expected}, got {projected}"
        );
    }

    /// Cylinder with axis perpendicular to flow: projected area = diameter * length.
    #[test]
    fn cylinder_projected_area_transverse_flow() {
        let r = 2.0;
        let l = 5.0;
        let shape = SurfaceShape::Cylinder {
            radius: r,
            length: l,
            axis: DVec3::Y,
        };
        let projected = shape.projected_area(DVec3::X);
        let expected = 2.0 * r * l;
        assert!(
            (projected - expected).abs() < TOL,
            "Transverse flow: expected {expected}, got {projected}"
        );
    }

    /// Cylinder at 45 degrees: area = pi*r^2*cos(45) + 2*r*L*sin(45).
    #[test]
    fn cylinder_projected_area_oblique() {
        let r = 2.0;
        let l = 5.0;
        let axis = DVec3::new(1.0, 1.0, 0.0).normalize();
        let shape = SurfaceShape::Cylinder {
            radius: r,
            length: l,
            axis,
        };
        let projected = shape.projected_area(DVec3::X);
        let c = FRAC_PI_4.cos();
        let s = FRAC_PI_4.sin();
        let expected = PI * r * r * c + 2.0 * r * l * s;
        assert!(
            (projected - expected).abs() < TOL,
            "45° cylinder: expected {expected}, got {projected}"
        );
    }

    /// Cylinder is symmetric under axis reversal.
    #[test]
    fn cylinder_projected_area_axis_symmetry() {
        let shape_plus = SurfaceShape::Cylinder {
            radius: 1.5,
            length: 4.0,
            axis: DVec3::X,
        };
        let shape_minus = SurfaceShape::Cylinder {
            radius: 1.5,
            length: 4.0,
            axis: -DVec3::X,
        };
        let v = DVec3::new(1.0, 0.5, 0.0).normalize();
        assert!(
            (shape_plus.projected_area(v) - shape_minus.projected_area(v)).abs() < TOL,
            "Cylinder should be symmetric under axis flip"
        );
    }

    // ── Conical frustum ───────────────────────────────────────────────

    /// Frustum with r_top == r_bottom == cylinder radius must match a
    /// cylinder of the same dimensions at all angles.
    #[test]
    fn cone_projected_area_degenerates_to_cylinder() {
        let r = 2.5;
        let l = 6.0;
        let axis = DVec3::Z;
        let cyl = SurfaceShape::Cylinder {
            radius: r,
            length: l,
            axis,
        };
        let frust = SurfaceShape::ConicalFrustum {
            radius_bottom: r,
            radius_top: r,
            length: l,
            axis,
        };
        for v in [DVec3::X, DVec3::Y, DVec3::Z, DVec3::new(1.0, 1.0, 1.0)] {
            let cyl_a = cyl.projected_area(v);
            let fr_a = frust.projected_area(v);
            assert!(
                (cyl_a - fr_a).abs() < TOL,
                "Degenerate frustum ≠ cylinder for v={v:?}: {cyl_a} vs {fr_a}"
            );
        }
    }

    /// Axial flow into the bottom cap of a frustum: projected area = pi*r_bottom^2.
    /// With axis = +X and flow = +X (v̂·axis > 0), the bottom cap is forward.
    #[test]
    fn cone_projected_area_axial_bottom_forward() {
        let rb = 3.0;
        let rt = 1.0;
        let l = 4.0;
        let shape = SurfaceShape::ConicalFrustum {
            radius_bottom: rb,
            radius_top: rt,
            length: l,
            axis: DVec3::X,
        };
        let projected = shape.projected_area(DVec3::X);
        let expected = PI * rb * rb;
        assert!(
            (projected - expected).abs() < TOL,
            "Axial (bottom forward): expected {expected}, got {projected}"
        );
    }

    /// Axial flow into the top cap (flow antiparallel to axis): forward cap
    /// is the top, area = pi*r_top^2.
    #[test]
    fn cone_projected_area_axial_top_forward() {
        let rb = 3.0;
        let rt = 1.0;
        let l = 4.0;
        let shape = SurfaceShape::ConicalFrustum {
            radius_bottom: rb,
            radius_top: rt,
            length: l,
            axis: DVec3::X,
        };
        let projected = shape.projected_area(-DVec3::X);
        let expected = PI * rt * rt;
        assert!(
            (projected - expected).abs() < TOL,
            "Axial (top forward): expected {expected}, got {projected}"
        );
    }

    /// Transverse flow: only the trapezoidal side contributes,
    /// area = (r_b + r_t) * L.
    #[test]
    fn cone_projected_area_transverse() {
        let rb = 3.0;
        let rt = 1.0;
        let l = 4.0;
        let shape = SurfaceShape::ConicalFrustum {
            radius_bottom: rb,
            radius_top: rt,
            length: l,
            axis: DVec3::X,
        };
        let projected = shape.projected_area(DVec3::Y);
        let expected = (rb + rt) * l;
        assert!(
            (projected - expected).abs() < TOL,
            "Transverse: expected {expected}, got {projected}"
        );
    }

    /// Oblique flow: verify the full frustum formula.
    #[test]
    fn cone_projected_area() {
        let rb = 3.0;
        let rt = 1.0;
        let l = 4.0;
        let shape = SurfaceShape::ConicalFrustum {
            radius_bottom: rb,
            radius_top: rt,
            length: l,
            axis: DVec3::X,
        };
        // Flow at 60° from axis: cos = 0.5, sin = sqrt(3)/2
        let v = DVec3::new(0.5, 3.0_f64.sqrt() / 2.0, 0.0);
        let projected = shape.projected_area(v);
        let cos_theta: f64 = 0.5;
        let sin_theta: f64 = 3.0_f64.sqrt() / 2.0;
        // cos_theta > 0 → bottom cap is forward
        let expected = (rb + rt) * l * sin_theta + PI * rb * rb * cos_theta.abs();
        assert!(
            (projected - expected).abs() < TOL,
            "Oblique frustum: expected {expected}, got {projected}"
        );
    }

    // ── Effective normal ──────────────────────────────────────────────

    /// Effective normal for a plate facing the flow is opposite to the flow.
    #[test]
    fn effective_normal_plate_faces_flow() {
        let shape = SurfaceShape::FlatPlate {
            area: 1.0,
            normal: DVec3::X, // plate normal points in +X, flow is +X
        };
        let eff = shape.effective_normal(DVec3::X);
        // Normal points with flow, so effective flips to -X
        assert!((eff - (-DVec3::X)).length() < TOL, "got {eff:?}");
    }

    /// Effective normal for a plate whose normal opposes the flow stays put.
    #[test]
    fn effective_normal_plate_back_faces_flow() {
        let shape = SurfaceShape::FlatPlate {
            area: 1.0,
            normal: -DVec3::X,
        };
        let eff = shape.effective_normal(DVec3::X);
        assert!((eff - (-DVec3::X)).length() < TOL);
    }

    /// Cylinder effective normal is along -flow regardless of axis.
    #[test]
    fn effective_normal_cylinder_along_flow() {
        let shape = SurfaceShape::Cylinder {
            radius: 1.0,
            length: 2.0,
            axis: DVec3::Y,
        };
        let v = DVec3::new(1.0, 0.0, 0.0);
        let eff = shape.effective_normal(v);
        assert!((eff - (-DVec3::X)).length() < TOL);
    }

    // ── Articulation ──────────────────────────────────────────────────

    /// Articulation rotates a flat plate's normal.
    #[test]
    fn articulation_rotates_normal() {
        let base = SurfaceFacet {
            shape: SurfaceShape::FlatPlate {
                area: 4.0,
                normal: DVec3::X,
            },
            center: DVec3::ZERO,
        };
        // Rotate 90° about +Z: +X → +Y
        let art = ArticulationState {
            axis: DVec3::Z,
            angle: std::f64::consts::FRAC_PI_2,
            rate: 0.0,
        };
        let facet = ArticulatedFacet::articulated(base, art);
        match facet.current_shape() {
            SurfaceShape::FlatPlate { normal, area } => {
                assert!((normal - DVec3::Y).length() < 1e-12, "got {normal:?}");
                assert_eq!(area, 4.0);
            }
            _ => panic!("expected FlatPlate"),
        }
    }

    /// Articulation rotates a cylinder's axis.
    #[test]
    fn articulation_rotates_cylinder_axis() {
        let base = SurfaceFacet {
            shape: SurfaceShape::Cylinder {
                radius: 1.0,
                length: 2.0,
                axis: DVec3::X,
            },
            center: DVec3::ZERO,
        };
        // Rotate 90° about +Z: +X → +Y
        let art = ArticulationState {
            axis: DVec3::Z,
            angle: std::f64::consts::FRAC_PI_2,
            rate: 0.0,
        };
        let facet = ArticulatedFacet::articulated(base, art);
        match facet.current_shape() {
            SurfaceShape::Cylinder {
                axis,
                radius,
                length,
            } => {
                assert!((axis - DVec3::Y).length() < 1e-12, "got {axis:?}");
                assert_eq!(radius, 1.0);
                assert_eq!(length, 2.0);
            }
            _ => panic!("expected Cylinder"),
        }
    }

    /// advance() propagates the angle by rate*dt.
    #[test]
    fn articulation_advance_integrates_rate() {
        let base = SurfaceFacet {
            shape: SurfaceShape::FlatPlate {
                area: 1.0,
                normal: DVec3::X,
            },
            center: DVec3::ZERO,
        };
        let art = ArticulationState {
            axis: DVec3::Z,
            angle: 0.0,
            rate: 0.1, // rad/s
        };
        let mut facet = ArticulatedFacet::articulated(base, art);
        facet.advance(5.0);
        assert!(
            (facet.articulation.unwrap().angle - 0.5).abs() < 1e-15,
            "angle = {}",
            facet.articulation.unwrap().angle
        );
    }

    /// A fixed facet is unaffected by advance() and current_shape() matches.
    #[test]
    fn fixed_facet_is_noop() {
        let base = SurfaceFacet {
            shape: SurfaceShape::FlatPlate {
                area: 4.0,
                normal: DVec3::X,
            },
            center: DVec3::ZERO,
        };
        let mut facet = ArticulatedFacet::fixed(base);
        facet.advance(100.0);
        assert_eq!(facet.current_shape(), base.shape);
    }

    // ── Edge cases ────────────────────────────────────────────────────

    /// Zero-length flow direction returns zero projected area.
    #[test]
    fn zero_flow_direction_zero_area() {
        let shape = SurfaceShape::FlatPlate {
            area: 10.0,
            normal: DVec3::X,
        };
        assert_eq!(shape.projected_area(DVec3::ZERO), 0.0);
        assert_eq!(shape.effective_normal(DVec3::ZERO), DVec3::ZERO);
    }

    /// Non-unit flow direction is normalized internally.
    #[test]
    fn flow_direction_magnitude_irrelevant() {
        let shape = SurfaceShape::Cylinder {
            radius: 2.0,
            length: 5.0,
            axis: DVec3::Y,
        };
        let a1 = shape.projected_area(DVec3::X);
        let a2 = shape.projected_area(DVec3::X * 1000.0);
        assert!((a1 - a2).abs() < TOL);
    }
}
