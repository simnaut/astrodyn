//! Structured cross-validation error reporting for Tier 3 tests.
//!
//! Tests create a [`CrossvalReport`], accumulate per-component max absolute
//! errors via [`CrossvalReport::accumulate`], then call [`CrossvalReport::write`]
//! to persist a JSON file to `target/tier3_crossval/<test_name>.json`.
//!
//! The report binary (`tier3_report`) reads all JSON files and generates
//! `target/tier3_report.md`.

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

/// Per-component max absolute errors and tolerances for a single test.
///
/// Standard state vector components (all per-axis max |Δ|):
/// - `position`: \[x, y, z\] in m
/// - `velocity`: \[x, y, z\] in m/s
/// - `acceleration`: \[x, y, z\] in m/s²
/// - `quaternion`: \[w, x, y, z\] component diffs + angle error in rad
/// - `ang_vel`: \[x, y, z\] body angular rates in rad/s
/// - `ang_accel`: \[x, y, z\] body angular accelerations in rad/s²
///
/// Each group has an optional per-component tolerance array.
/// Test-specific extras are appended as named `(var, val, tol, unit)` tuples.
pub struct CrossvalReport {
    test_name: String,

    // Max absolute errors per component
    pub position: Option<[f64; 3]>,
    pub velocity: Option<[f64; 3]>,
    pub acceleration: Option<[f64; 3]>,
    pub quaternion: Option<[f64; 4]>,
    pub quat_angle: Option<f64>,
    pub ang_vel: Option<[f64; 3]>,
    pub ang_accel: Option<[f64; 3]>,

    // Per-component tolerances (None = no threshold)
    pub position_tol: Option<[f64; 3]>,
    pub velocity_tol: Option<[f64; 3]>,
    pub acceleration_tol: Option<[f64; 3]>,
    pub quaternion_tol: Option<[f64; 4]>,
    pub quat_angle_tol: Option<f64>,
    pub ang_vel_tol: Option<[f64; 3]>,
    pub ang_accel_tol: Option<[f64; 3]>,

    // Test-specific extras
    extras: Vec<(String, f64, Option<f64>, String)>,
}

/// Snapshot of state at a single timestep for comparison.
#[derive(Default)]
pub struct StateSnapshot {
    pub position: Option<DVec3>,
    pub velocity: Option<DVec3>,
    pub acceleration: Option<DVec3>,
    pub quaternion: Option<DQuat>,
    pub ang_vel: Option<DVec3>,
    pub ang_accel: Option<DVec3>,
}

impl CrossvalReport {
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            position: None,
            velocity: None,
            acceleration: None,
            quaternion: None,
            quat_angle: None,
            ang_vel: None,
            ang_accel: None,
            position_tol: None,
            velocity_tol: None,
            acceleration_tol: None,
            quaternion_tol: None,
            quat_angle_tol: None,
            ang_vel_tol: None,
            ang_accel_tol: None,
            extras: Vec::new(),
        }
    }

    /// Accumulate per-component max absolute errors from two state snapshots.
    pub fn accumulate(&mut self, ours: &StateSnapshot, reference: &StateSnapshot) {
        if let (Some(a), Some(b)) = (ours.position, reference.position) {
            let d = a - b;
            let e = self.position.get_or_insert([0.0; 3]);
            e[0] = e[0].max(d.x.abs());
            e[1] = e[1].max(d.y.abs());
            e[2] = e[2].max(d.z.abs());
        }
        if let (Some(a), Some(b)) = (ours.velocity, reference.velocity) {
            let d = a - b;
            let e = self.velocity.get_or_insert([0.0; 3]);
            e[0] = e[0].max(d.x.abs());
            e[1] = e[1].max(d.y.abs());
            e[2] = e[2].max(d.z.abs());
        }
        if let (Some(a), Some(b)) = (ours.acceleration, reference.acceleration) {
            let d = a - b;
            let e = self.acceleration.get_or_insert([0.0; 3]);
            e[0] = e[0].max(d.x.abs());
            e[1] = e[1].max(d.y.abs());
            e[2] = e[2].max(d.z.abs());
        }
        if let (Some(a), Some(b)) = (ours.quaternion, reference.quaternion) {
            // Per-component diffs
            let e = self.quaternion.get_or_insert([0.0; 4]);
            e[0] = e[0].max((a.w - b.w).abs());
            e[1] = e[1].max((a.x - b.x).abs());
            e[2] = e[2].max((a.y - b.y).abs());
            e[3] = e[3].max((a.z - b.z).abs());
            // Angle error: acos(2(q1·q2)² - 1)
            let dot = (a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z).abs();
            let angle = (2.0 * dot * dot - 1.0).clamp(-1.0, 1.0).acos();
            let qa = self.quat_angle.get_or_insert(0.0);
            *qa = qa.max(angle);
        }
        if let (Some(a), Some(b)) = (ours.ang_vel, reference.ang_vel) {
            let d = a - b;
            let e = self.ang_vel.get_or_insert([0.0; 3]);
            e[0] = e[0].max(d.x.abs());
            e[1] = e[1].max(d.y.abs());
            e[2] = e[2].max(d.z.abs());
        }
        if let (Some(a), Some(b)) = (ours.ang_accel, reference.ang_accel) {
            let d = a - b;
            let e = self.ang_accel.get_or_insert([0.0; 3]);
            e[0] = e[0].max(d.x.abs());
            e[1] = e[1].max(d.y.abs());
            e[2] = e[2].max(d.z.abs());
        }
    }

    /// Add a test-specific extra metric.
    /// Use `f64::INFINITY` for tolerance when there is no explicit threshold.
    pub fn add_extra(&mut self, var: &str, val: f64, tol: f64, unit: &str) {
        let tol = if tol.is_finite() { Some(tol) } else { None };
        self.extras
            .push((var.to_string(), val, tol, unit.to_string()));
    }

    /// Write the report to `target/tier3_crossval/<test_name>.json`.
    pub fn write(&self) {
        let dir = output_dir();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("{}.json", self.test_name));

        let mut json = format!(r#"{{"test":"{}""#, self.test_name);

        write_vec3_field(&mut json, "position", &self.position);
        write_vec3_field(&mut json, "velocity", &self.velocity);
        write_vec3_field(&mut json, "acceleration", &self.acceleration);
        write_vec4_field(&mut json, "quaternion", &self.quaternion);
        write_f64_field(&mut json, "quat_angle", &self.quat_angle);
        write_vec3_field(&mut json, "ang_vel", &self.ang_vel);
        write_vec3_field(&mut json, "ang_accel", &self.ang_accel);

        write_vec3_field(&mut json, "position_tol", &self.position_tol);
        write_vec3_field(&mut json, "velocity_tol", &self.velocity_tol);
        write_vec3_field(&mut json, "acceleration_tol", &self.acceleration_tol);
        write_vec4_field(&mut json, "quaternion_tol", &self.quaternion_tol);
        write_f64_field(&mut json, "quat_angle_tol", &self.quat_angle_tol);
        write_vec3_field(&mut json, "ang_vel_tol", &self.ang_vel_tol);
        write_vec3_field(&mut json, "ang_accel_tol", &self.ang_accel_tol);

        // Extras
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
                json.push_str(&format!(
                    r#"{{"var":"{var}","val":{val:.6e},"tol":{tol_str},"unit":"{unit}"}}"#
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

/// Legacy wrapper — emits extras-only JSON for tests not yet converted to
/// `CrossvalReport`. Will be removed once all tests are migrated.
#[deprecated(note = "Use CrossvalReport::new() + accumulate() + write() instead")]
pub fn crossval_report(test_name: &str, metrics: &[(&str, f64, f64, &str)]) {
    let mut report = CrossvalReport::new(test_name);
    for (var, val, tol, unit) in metrics {
        report.add_extra(var, *val, *tol, unit);
    }
    report.write();
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
