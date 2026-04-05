//! Structured cross-validation error reporting for Tier 3 tests.
//!
//! Tests log their propagated state at each timestep, then call
//! [`CrossvalReport::compute`] with the reference trajectory to produce
//! per-component max absolute errors. The report is written to
//! `target/tier3_crossval/<test_name>.json`.

use glam::{DQuat, DVec3};
use std::io::Write;
use std::path::PathBuf;

fn output_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            break;
        }
        if !dir.pop() {
            dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            break;
        }
    }
    dir.join("target").join("tier3_crossval")
}

/// A single state snapshot at one timestep.
#[derive(Clone, Default)]
pub struct StateLog {
    pub time: f64,
    pub position: Option<DVec3>,
    pub velocity: Option<DVec3>,
    pub acceleration: Option<DVec3>,
    pub quaternion: Option<DQuat>,
    pub ang_vel: Option<DVec3>,
    pub ang_accel: Option<DVec3>,
}

/// Per-component max absolute errors, plus test-specific extras.
///
/// Tolerances live exclusively in the test source code (assert statements).
/// The report binary extracts them from source for display.
pub struct CrossvalReport {
    test_name: String,

    // Computed by `compute()` — per-component max |ours - ref| across trajectory
    pub position: Option<[f64; 3]>,
    pub velocity: Option<[f64; 3]>,
    pub acceleration: Option<[f64; 3]>,
    pub quaternion: Option<[f64; 4]>,
    pub quat_angle: Option<f64>,
    pub ang_vel: Option<[f64; 3]>,
    pub ang_accel: Option<[f64; 3]>,

    // Test-specific extras: (variable_name, value, tolerance, unit)
    // Tolerance is kept here temporarily for the report binary transition.
    extras: Vec<(String, f64, Option<f64>, String)>,
}

impl CrossvalReport {
    /// Compute per-component max absolute errors from two trajectories.
    ///
    /// `ours` and `reference` must be the same length and time-aligned.
    pub fn compute(test_name: &str, ours: &[StateLog], reference: &[StateLog]) -> Self {
        assert_eq!(
            ours.len(),
            reference.len(),
            "Trajectory lengths differ: ours={}, ref={}",
            ours.len(),
            reference.len()
        );

        let mut report = Self {
            test_name: test_name.to_string(),
            position: None,
            velocity: None,
            acceleration: None,
            quaternion: None,
            quat_angle: None,
            ang_vel: None,
            ang_accel: None,
            extras: Vec::new(),
        };

        for (a, b) in ours.iter().zip(reference.iter()) {
            if let (Some(av), Some(bv)) = (a.position, b.position) {
                let d = av - bv;
                let e = report.position.get_or_insert([0.0; 3]);
                e[0] = e[0].max(d.x.abs());
                e[1] = e[1].max(d.y.abs());
                e[2] = e[2].max(d.z.abs());
            }
            if let (Some(av), Some(bv)) = (a.velocity, b.velocity) {
                let d = av - bv;
                let e = report.velocity.get_or_insert([0.0; 3]);
                e[0] = e[0].max(d.x.abs());
                e[1] = e[1].max(d.y.abs());
                e[2] = e[2].max(d.z.abs());
            }
            if let (Some(av), Some(bv)) = (a.acceleration, b.acceleration) {
                let d = av - bv;
                let e = report.acceleration.get_or_insert([0.0; 3]);
                e[0] = e[0].max(d.x.abs());
                e[1] = e[1].max(d.y.abs());
                e[2] = e[2].max(d.z.abs());
            }
            if let (Some(aq), Some(bq)) = (a.quaternion, b.quaternion) {
                // Canonicalize: q and -q represent the same rotation.
                // If dot < 0, negate one so component diffs are meaningful.
                let dot = aq.w * bq.w + aq.x * bq.x + aq.y * bq.y + aq.z * bq.z;
                let bq = if dot < 0.0 {
                    DQuat::from_xyzw(-bq.x, -bq.y, -bq.z, -bq.w)
                } else {
                    bq
                };
                let e = report.quaternion.get_or_insert([0.0; 4]);
                e[0] = e[0].max((aq.w - bq.w).abs());
                e[1] = e[1].max((aq.x - bq.x).abs());
                e[2] = e[2].max((aq.y - bq.y).abs());
                e[3] = e[3].max((aq.z - bq.z).abs());
                let angle = (2.0 * dot.abs() * dot.abs() - 1.0).clamp(-1.0, 1.0).acos();
                let qa = report.quat_angle.get_or_insert(0.0);
                *qa = qa.max(angle);
            }
            if let (Some(av), Some(bv)) = (a.ang_vel, b.ang_vel) {
                let d = av - bv;
                let e = report.ang_vel.get_or_insert([0.0; 3]);
                e[0] = e[0].max(d.x.abs());
                e[1] = e[1].max(d.y.abs());
                e[2] = e[2].max(d.z.abs());
            }
            if let (Some(av), Some(bv)) = (a.ang_accel, b.ang_accel) {
                let d = av - bv;
                let e = report.ang_accel.get_or_insert([0.0; 3]);
                e[0] = e[0].max(d.x.abs());
                e[1] = e[1].max(d.y.abs());
                e[2] = e[2].max(d.z.abs());
            }
        }

        report
    }

    /// Add a test-specific extra metric.
    /// Use `f64::INFINITY` for tolerance when there is no explicit threshold.
    pub fn add_extra(&mut self, var: &str, val: f64, tol: f64, unit: &str) {
        let tol = if tol.is_finite() { Some(tol) } else { None };
        self.extras
            .push((var.to_string(), val, tol, unit.to_string()));
    }

    /// Worst-component position error (∞-norm, for assert! statements).
    pub fn max_position_component(&self) -> f64 {
        self.position
            .map(|p| p.iter().copied().fold(0.0_f64, f64::max))
            .unwrap_or(0.0)
    }

    /// Worst-component velocity error (∞-norm, for assert! statements).
    pub fn max_velocity_component(&self) -> f64 {
        self.velocity
            .map(|v| v.iter().copied().fold(0.0_f64, f64::max))
            .unwrap_or(0.0)
    }

    /// Worst-component angular velocity error (∞-norm, for assert! statements).
    pub fn max_ang_vel_component(&self) -> f64 {
        self.ang_vel
            .map(|v| v.iter().copied().fold(0.0_f64, f64::max))
            .unwrap_or(0.0)
    }

    /// Quaternion angle error in radians (rotation-invariant).
    pub fn max_quat_angle(&self) -> f64 {
        self.quat_angle.unwrap_or(0.0)
    }

    /// Assert each position component is within its tolerance.
    pub fn assert_position(&self, tol: [f64; 3]) {
        let p = self.position.expect("no position data");
        for (i, label) in ["x", "y", "z"].iter().enumerate() {
            assert!(
                p[i] < tol[i],
                "{}: position_{label} error {:.6e} m exceeds tolerance {:.6e} m",
                self.test_name,
                p[i],
                tol[i]
            );
        }
    }

    /// Assert each velocity component is within its tolerance.
    pub fn assert_velocity(&self, tol: [f64; 3]) {
        let v = self.velocity.expect("no velocity data");
        for (i, label) in ["x", "y", "z"].iter().enumerate() {
            assert!(
                v[i] < tol[i],
                "{}: velocity_{label} error {:.6e} m/s exceeds tolerance {:.6e} m/s",
                self.test_name,
                v[i],
                tol[i]
            );
        }
    }

    /// Assert quaternion angle error is within tolerance.
    pub fn assert_quat_angle(&self, tol: f64) {
        let q = self.quat_angle.expect("no quaternion angle data");
        assert!(
            q < tol,
            "{}: quat_angle error {:.6e} rad exceeds tolerance {:.6e} rad",
            self.test_name,
            q,
            tol
        );
    }

    /// Assert each angular velocity component is within its tolerance.
    pub fn assert_ang_vel(&self, tol: [f64; 3]) {
        let w = self.ang_vel.expect("no angular velocity data");
        for (i, label) in ["x", "y", "z"].iter().enumerate() {
            assert!(
                w[i] < tol[i],
                "{}: ang_vel_{label} error {:.6e} rad/s exceeds tolerance {:.6e} rad/s",
                self.test_name,
                w[i],
                tol[i]
            );
        }
    }

    /// Write the report to `target/tier3_crossval/<test_name>.json`.
    pub fn write(&self) {
        let dir = output_dir();
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| {
            panic!(
                "failed to create tier3_crossval directory {}: {e}",
                dir.display()
            )
        });
        let path = dir.join(format!("{}.json", self.test_name));

        let mut json = format!(r#"{{"test":"{}""#, json_escape(&self.test_name));

        write_vec3_field(&mut json, "position", &self.position);
        write_vec3_field(&mut json, "velocity", &self.velocity);
        write_vec3_field(&mut json, "acceleration", &self.acceleration);
        write_vec4_field(&mut json, "quaternion", &self.quaternion);
        write_f64_field(&mut json, "quat_angle", &self.quat_angle);
        write_vec3_field(&mut json, "ang_vel", &self.ang_vel);
        write_vec3_field(&mut json, "ang_accel", &self.ang_accel);

        if !self.extras.is_empty() {
            json.push_str(",\"extras\":[");
            for (i, (var, val, tol, unit)) in self.extras.iter().enumerate() {
                if i > 0 {
                    json.push(',');
                }
                let tol_str = match tol {
                    Some(t) => format!("{t:.6e}"),
                    None => "null".to_string(),
                };
                let var_esc = json_escape(var);
                let unit_esc = json_escape(unit);
                json.push_str(&format!(
                    r#"{{"var":"{var_esc}","val":{val:.6e},"tol":{tol_str},"unit":"{unit_esc}"}}"#
                ));
            }
            json.push(']');
        }

        json.push('}');

        let mut file =
            std::fs::File::create(&path).expect("failed to create tier3_crossval JSON file");
        file.write_all(json.as_bytes())
            .expect("failed to write tier3_crossval JSON file");
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn write_vec3_field(json: &mut String, name: &str, val: &Option<[f64; 3]>) {
    match val {
        Some([x, y, z]) => {
            json.push_str(&format!(r#","{name}":[{x:.6e},{y:.6e},{z:.6e}]"#));
        }
        None => {
            json.push_str(&format!(r#","{name}":null"#));
        }
    }
}

fn write_vec4_field(json: &mut String, name: &str, val: &Option<[f64; 4]>) {
    match val {
        Some([w, x, y, z]) => {
            json.push_str(&format!(r#","{name}":[{w:.6e},{x:.6e},{y:.6e},{z:.6e}]"#));
        }
        None => {
            json.push_str(&format!(r#","{name}":null"#));
        }
    }
}

fn write_f64_field(json: &mut String, name: &str, val: &Option<f64>) {
    match val {
        Some(v) => {
            json.push_str(&format!(r#","{name}":{v:.6e}"#));
        }
        None => {
            json.push_str(&format!(r#","{name}":null"#));
        }
    }
}
