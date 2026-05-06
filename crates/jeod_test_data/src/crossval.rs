//! Structured cross-validation error reporting for Tier 3 tests.
//!
//! Tests log their propagated state at each timestep, then call
//! [`CrossvalReport::compute`] with the reference trajectory to produce
//! per-component max absolute errors. The report is written to
//! `target/tier3_crossval/<test_name>.json`.

use glam::{DQuat, DVec3};
use jeod_quantities::ext::F64Ext;
use std::io::Write;
use std::path::PathBuf;
use uom::si::f64::{Angle, AngularVelocity, Length, Velocity};

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

/// Estimate the reference trajectory's nominal sample cadence as the median
/// gap between consecutive timestamps. Falls back to `1.0` for trajectories
/// shorter than two samples (alignment is trivial in that case anyway).
///
/// For an odd number of gaps the median is the unique middle element. For an
/// even number it is the mean of the two middle elements — picking the
/// upper-middle naively would bias toward the larger gap and loosen the
/// alignment tolerance for non-uniform sampling.
fn reference_timestep(reference: &[StateLog]) -> f64 {
    if reference.len() < 2 {
        return 1.0;
    }
    let mut gaps: Vec<f64> = reference
        .windows(2)
        .map(|w| (w[1].time - w[0].time).abs())
        .collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = gaps.len();
    let median = if n % 2 == 1 {
        gaps[n / 2]
    } else {
        // Average of the two middle elements.
        (gaps[n / 2 - 1] + gaps[n / 2]) * 0.5
    };
    median.max(f64::EPSILON)
}

/// A single state snapshot at one timestep.
#[derive(Clone, Default)]
pub struct StateLog {
    /// Sample time in seconds since the trajectory's t=0.
    pub time: f64,
    /// Position in metres, when sampled.
    pub position: Option<DVec3>,
    /// Velocity in m/s, when sampled.
    pub velocity: Option<DVec3>,
    /// Acceleration in m/s², when sampled.
    pub acceleration: Option<DVec3>,
    /// Attitude quaternion, when sampled.
    pub quaternion: Option<DQuat>,
    /// Angular velocity in rad/s, when sampled.
    pub ang_vel: Option<DVec3>,
    /// Angular acceleration in rad/s², when sampled.
    pub ang_accel: Option<DVec3>,
}

/// Per-component max absolute errors, plus test-specific extras.
///
/// Tolerances live exclusively in the test source code (assert statements).
/// The report binary extracts them from source for display.
pub struct CrossvalReport {
    test_name: String,

    /// Per-axis max `|ours − ref|` of position over the trajectory.
    pub position: Option<[f64; 3]>,
    /// Per-axis max `|ours − ref|` of velocity over the trajectory.
    pub velocity: Option<[f64; 3]>,
    /// Per-axis max `|ours − ref|` of acceleration over the trajectory.
    pub acceleration: Option<[f64; 3]>,
    /// Per-component max `|ours − ref|` of the quaternion (`[x,y,z,w]`).
    pub quaternion: Option<[f64; 4]>,
    /// Maximum attitude error angle (radians) inferred from the
    /// quaternion delta.
    pub quat_angle: Option<f64>,
    /// Per-axis max `|ours − ref|` of angular velocity.
    pub ang_vel: Option<[f64; 3]>,
    /// Per-axis max `|ours − ref|` of angular acceleration.
    pub ang_accel: Option<[f64; 3]>,

    // Test-specific extras: (variable_name, value, unit)
    extras: Vec<(String, f64, String)>,
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

        // Verify time alignment. Tolerance scales with the reference's own
        // sample cadence so sub-100 ms cadences (drag at 1 s, contact at
        // 0.01 s) still get a tight check; reference rows that drift by more
        // than 10 % of the reference timestep would silently fall onto an
        // adjacent sample under a flat tolerance.
        let ref_step = reference_timestep(reference);
        let alignment_tolerance = 0.1 * ref_step;
        for (i, (a, b)) in ours.iter().zip(reference.iter()).enumerate() {
            let dt = (a.time - b.time).abs();
            assert!(
                dt < alignment_tolerance,
                "Time mismatch at index {i}: ours={:.3}s, ref={:.3}s (delta={dt:.6e}s, tol={alignment_tolerance:.6e}s)",
                a.time,
                b.time
            );
        }

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

    /// Add a test-specific extra metric (error value only, no tolerance).
    /// Tolerances live in the test's `assert!` statements.
    pub fn add_extra(&mut self, var: &str, val: f64, unit: &str) {
        self.extras.push((var.to_string(), val, unit.to_string()));
    }

    /// Assert that every reference-CSV sample time falls on an integrator
    /// output instant — i.e., that `row.time` is an integer multiple of
    /// `integrator_dt` within a small tolerance.
    ///
    /// Tier 3 cross-validation compares our trajectory against JEOD's CSV
    /// row-by-row. When JEOD's logger writes faster than the integrator
    /// runs (e.g. CSV at 0.5 s while `IntegLoop ... DYNAMICS=1.0`), Trick
    /// holds and re-emits the integrator's output from the previous
    /// integer second on the off-cadence rows. A naive row-by-row
    /// comparison passes vacuously on those held rows because both sides
    /// quote the same earlier state, masking real residuals at the actual
    /// integrator-output instants.
    ///
    /// Call this at the top of a Tier 3 test to fail loudly when the
    /// integrator step does not divide the CSV cadence. If a test is
    /// deliberately running an off-cadence integrator (e.g. dt=1.0 s
    /// against a 0.5 s CSV), it must filter the off-cadence rows out
    /// before logging — see [`CrossvalReport::is_on_integrator_cadence`]
    /// for the per-row helper that the existing
    /// `tier3_sim_ref_attach.rs` half-second skip pattern uses — and
    /// then call `assert_cadence_matches` on the *filtered* sample
    /// times so the helper still sees a clean cadence-aligned set.
    ///
    /// `tolerance_fraction` is the fraction of `integrator_dt` allowed
    /// as f64 round-off slack on each row's modular residual; `1e-6` is
    /// the conservative default used at the existing per-test skip
    /// sites.
    pub fn assert_cadence_matches(
        reference: &[StateLog],
        integrator_dt: f64,
        tolerance_fraction: f64,
    ) {
        Self::assert_cadence_matches_times(
            reference.iter().map(|s| s.time),
            integrator_dt,
            tolerance_fraction,
        );
    }

    /// Same as [`CrossvalReport::assert_cadence_matches`] but takes a raw
    /// iterator of CSV row times. Use this when the test holds the CSV
    /// rows in a domain-specific record type (e.g. `DyncompRecord`)
    /// rather than `StateLog`, so the cadence check can run before any
    /// `StateLog` is built — turning the cadence assertion into the very
    /// first thing the row loop sees.
    pub fn assert_cadence_matches_times<I>(times: I, integrator_dt: f64, tolerance_fraction: f64)
    where
        I: IntoIterator<Item = f64>,
    {
        assert!(
            integrator_dt > 0.0,
            "assert_cadence_matches: integrator_dt must be positive, got {integrator_dt}"
        );
        assert!(
            (0.0..1.0).contains(&tolerance_fraction),
            "assert_cadence_matches: tolerance_fraction must be in [0, 1), got {tolerance_fraction}"
        );
        let abs_tol = (tolerance_fraction * integrator_dt).max(f64::EPSILON);
        for (i, t) in times.into_iter().enumerate() {
            let n = (t / integrator_dt).round();
            let modular_err = (t - n * integrator_dt).abs();
            assert!(
                modular_err <= abs_tol,
                "Tier 3 cadence mismatch at reference row {i} (t = {t:.9} s): \
                 not an integer multiple of integrator_dt = {integrator_dt} s \
                 (closest multiple = {:.9} s, residual = {modular_err:.3e} s, \
                 tolerance = {abs_tol:.3e} s). \
                 Either change the integrator dt to divide the CSV cadence, or \
                 filter off-cadence rows out of `reference` before calling \
                 `CrossvalReport::compute` — see `is_on_integrator_cadence` and \
                 the half-second-skip pattern in \
                 `crates/jeod_runner/tests/tier3_sim_ref_attach.rs`.",
                n * integrator_dt
            );
        }
    }

    /// Per-row predicate: `true` iff `row_time` is an integer multiple of
    /// `integrator_dt` within `1e-6 * integrator_dt` round-off slack.
    ///
    /// Use inside the test's main row loop as the standard cadence skip:
    ///
    /// ```ignore
    /// for row in &rows {
    ///     if !CrossvalReport::is_on_integrator_cadence(row.time, dt) {
    ///         continue;  // off-cadence sample, Trick re-emits the prior step's state
    ///     }
    ///     // ...compare this row...
    /// }
    /// ```
    ///
    /// This is the catch-able tooling form of the
    /// `(row.time - row.time.round()).abs() > 1e-6` skip already used in
    /// `tier3_sim_ref_attach.rs` (whose dt happens to be 1.0 s, so
    /// `.round()` and "integer multiple of dt" coincide).
    pub fn is_on_integrator_cadence(row_time: f64, integrator_dt: f64) -> bool {
        assert!(
            integrator_dt > 0.0,
            "is_on_integrator_cadence: integrator_dt must be positive, got {integrator_dt}"
        );
        let n = (row_time / integrator_dt).round();
        let abs_tol = (1e-6 * integrator_dt).max(f64::EPSILON);
        (row_time - n * integrator_dt).abs() <= abs_tol
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

    /// Worst-component position error as a typed [`Length`] (meters).
    pub fn max_position_typed(&self) -> Length {
        self.max_position_component().m()
    }

    /// Worst-component velocity error as a typed [`Velocity`] (m/s).
    pub fn max_velocity_typed(&self) -> Velocity {
        self.max_velocity_component().m_per_s()
    }

    /// Worst-component angular velocity error as a typed [`AngularVelocity`] (rad/s).
    pub fn max_ang_vel_typed(&self) -> AngularVelocity {
        self.max_ang_vel_component().rad_per_s()
    }

    /// Quaternion angle error as a typed [`Angle`] (radians).
    pub fn max_quat_angle_typed(&self) -> Angle {
        self.max_quat_angle().rad()
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
            for (i, (var, val, unit)) in self.extras.iter().enumerate() {
                if i > 0 {
                    json.push(',');
                }
                let var_esc = json_escape(var);
                let unit_esc = json_escape(unit);
                json.push_str(&format!(
                    r#"{{"var":"{var_esc}","val":{val:.6e},"unit":"{unit_esc}"}}"#
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

/// Escape a string for inclusion in a JSON literal. Pub so the `tier3_report`
/// binary (and any other consumer) can produce valid JSON without re-rolling
/// the escape rules.
pub fn json_escape(s: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_log_at(times: &[f64]) -> Vec<StateLog> {
        times
            .iter()
            .map(|&t| StateLog {
                time: t,
                ..StateLog::default()
            })
            .collect()
    }

    #[test]
    fn cadence_matches_when_csv_is_integer_multiple_of_dt() {
        // 60 s CSV cadence with a 0.03125 s integrator step (32 Hz):
        // 60.0 / 0.03125 = 1920 — integer. Mirrors
        // `tier3_sim_dyncomp_run_attach_to_ref_frame.rs`.
        let times: Vec<f64> = (0..201).map(|i| i as f64 * 60.0).collect();
        let reference = ref_log_at(&times);
        CrossvalReport::assert_cadence_matches(&reference, 0.03125, 1e-6);
    }

    #[test]
    fn cadence_matches_when_csv_equals_dt() {
        // Apollo 8 test: dt=0.5 s and CSV at 0.5 s.
        let times: Vec<f64> = (0..201).map(|i| i as f64 * 0.5).collect();
        let reference = ref_log_at(&times);
        CrossvalReport::assert_cadence_matches(&reference, 0.5, 1e-6);
    }

    #[test]
    #[should_panic(expected = "Tier 3 cadence mismatch")]
    fn cadence_panics_on_half_second_csv_with_one_second_integrator() {
        // The exact bug shape #355 targets: CSV samples at 0.5 s but
        // the integrator runs at 1.0 s. Half-second rows are not on
        // any integrator-output instant.
        let times: Vec<f64> = (0..101).map(|i| i as f64 * 0.5).collect();
        let reference = ref_log_at(&times);
        CrossvalReport::assert_cadence_matches(&reference, 1.0, 1e-6);
    }

    #[test]
    fn is_on_cadence_filters_half_seconds_at_dt_one() {
        // The half-second rows must skip; the integer-second rows pass.
        assert!(CrossvalReport::is_on_integrator_cadence(0.0, 1.0));
        assert!(CrossvalReport::is_on_integrator_cadence(50.0, 1.0));
        assert!(!CrossvalReport::is_on_integrator_cadence(0.5, 1.0));
        assert!(!CrossvalReport::is_on_integrator_cadence(50.5, 1.0));
    }

    #[test]
    fn is_on_cadence_handles_f64_jitter_within_one_ppm_of_dt() {
        // f64 round-off well below 1e-6 * dt should still register as
        // on-cadence.
        assert!(CrossvalReport::is_on_integrator_cadence(
            60.0 + 1e-12,
            0.03125
        ));
    }
}
