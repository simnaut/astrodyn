//! Tier 3: Cross-validate orbital elements computation against JEOD SIM_OrbElem RUN_ecc.
//!
//! At each timestep, reads position and velocity from the JEOD CSV, computes
//! `OrbitalElements::from_cartesian()`, and compares every element field against
//! the JEOD-logged values.
//!
//! Requires Docker-generated CSV (see test_data/README.md).

use glam::DVec3;
use jeod_math::OrbitalElements;
use std::path::Path;

/// Earth gravitational parameter (m^3/s^2), matching JEOD's value.
const MU_EARTH: f64 = 3.986_004_418e14;

/// Parsed record from the SIM_OrbElem CSV.
#[derive(Debug)]
struct OrbElemRecord {
    time: f64,
    semi_major_axis: f64,
    semiparam: f64,
    e_mag: f64,
    inclination: f64,
    arg_periapsis: f64,
    long_asc_node: f64,
    r_mag: f64,
    vel_mag: f64,
    true_anom: f64,
    mean_anom: f64,
    mean_motion: f64,
    orbital_anom: f64,
    orb_energy: f64,
    orb_ang_momentum: f64,
    position: DVec3,
    velocity: DVec3,
}

fn load_orbelem_csv(path: &Path) -> Vec<OrbElemRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_OrbElem CSV from {}: {e}",
            path.display()
        )
    });

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 21 {
            continue;
        }

        let line_no = i + 1;
        let parse = |col: usize| -> f64 {
            fields[col].trim().parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "Failed to parse OrbElem CSV at line {line_no}, col {col}: {:?} ({e})",
                    fields[col]
                )
            })
        };

        // CSV columns:
        // 0: time
        // 1: semi_major_axis, 2: semiparam, 3: e_mag, 4: inclination,
        // 5: arg_periapsis, 6: long_asc_node, 7: r_mag, 8: vel_mag,
        // 9: true_anom, 10: mean_anom, 11: mean_motion, 12: orbital_anom,
        // 13: orb_energy, 14: orb_ang_momentum,
        // 15: position[0], 16: position[1], 17: position[2],
        // 18: velocity[0], 19: velocity[1], 20: velocity[2]
        records.push(OrbElemRecord {
            time: parse(0),
            semi_major_axis: parse(1),
            semiparam: parse(2),
            e_mag: parse(3),
            inclination: parse(4),
            arg_periapsis: parse(5),
            long_asc_node: parse(6),
            r_mag: parse(7),
            vel_mag: parse(8),
            true_anom: parse(9),
            mean_anom: parse(10),
            mean_motion: parse(11),
            orbital_anom: parse(12),
            orb_energy: parse(13),
            orb_ang_momentum: parse(14),
            position: DVec3::new(parse(15), parse(16), parse(17)),
            velocity: DVec3::new(parse(18), parse(19), parse(20)),
        });
    }
    records
}

/// Compute angular difference accounting for wraparound at 2*pi.
fn angle_diff(a: f64, b: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    let mut d = (a - b) % tau;
    if d > std::f64::consts::PI {
        d -= tau;
    }
    if d < -std::f64::consts::PI {
        d += tau;
    }
    d.abs()
}

#[test]
#[ignore = "requires Docker-generated CSV — see test_data/README.md"]
fn tier3_orbital_elements_vs_jeod_sim_orbelem() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/orbelem_ecc_orbelem_ASCII.csv");

    assert!(
        csv_path.exists(),
        "SIM_OrbElem RUN_ecc CSV not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let records = load_orbelem_csv(&csv_path);
    assert!(
        records.len() > 10,
        "Expected more than 10 records in OrbElem CSV, got {}",
        records.len()
    );

    eprintln!(
        "Tier 3: SIM_OrbElem RUN_ecc cross-validation ({} timesteps)",
        records.len()
    );

    let mut max_sma_err = 0.0_f64;
    let mut max_ecc_err = 0.0_f64;
    let mut max_inc_err = 0.0_f64;
    let mut max_aop_err = 0.0_f64;
    let mut max_lan_err = 0.0_f64;
    let mut max_ta_err = 0.0_f64;
    let mut max_ma_err = 0.0_f64;
    let mut max_mm_err = 0.0_f64;
    let mut max_oa_err = 0.0_f64;
    let mut max_energy_err = 0.0_f64;
    let mut max_hmag_err = 0.0_f64;
    let mut max_rmag_err = 0.0_f64;
    let mut max_vmag_err = 0.0_f64;
    let mut max_sp_err = 0.0_f64;

    for (idx, rec) in records.iter().enumerate() {
        let oe = OrbitalElements::from_cartesian(MU_EARTH, rec.position, rec.velocity)
            .unwrap_or_else(|e| {
                panic!(
                    "from_cartesian failed at t={:.1}s (record {}): {e}",
                    rec.time, idx
                )
            });

        let sma_err = (oe.semi_major_axis - rec.semi_major_axis).abs();
        let sp_err = (oe.semiparam - rec.semiparam).abs();
        let ecc_err = (oe.e_mag - rec.e_mag).abs();
        let inc_err = (oe.inclination - rec.inclination).abs();
        let aop_err = angle_diff(oe.arg_periapsis, rec.arg_periapsis);
        let lan_err = angle_diff(oe.long_asc_node, rec.long_asc_node);
        let ta_err = angle_diff(oe.true_anom, rec.true_anom);
        let ma_err = angle_diff(oe.mean_anom, rec.mean_anom);
        let mm_err = (oe.mean_motion - rec.mean_motion).abs();
        let oa_err = angle_diff(oe.orbital_anom, rec.orbital_anom);
        let energy_err = (oe.orb_energy - rec.orb_energy).abs();
        let hmag_err = (oe.orb_ang_momentum - rec.orb_ang_momentum).abs();
        let rmag_err = (oe.r_mag - rec.r_mag).abs();
        let vmag_err = (oe.vel_mag - rec.vel_mag).abs();

        max_sma_err = max_sma_err.max(sma_err);
        max_sp_err = max_sp_err.max(sp_err);
        max_ecc_err = max_ecc_err.max(ecc_err);
        max_inc_err = max_inc_err.max(inc_err);
        max_aop_err = max_aop_err.max(aop_err);
        max_lan_err = max_lan_err.max(lan_err);
        max_ta_err = max_ta_err.max(ta_err);
        max_ma_err = max_ma_err.max(ma_err);
        max_mm_err = max_mm_err.max(mm_err);
        max_oa_err = max_oa_err.max(oa_err);
        max_energy_err = max_energy_err.max(energy_err);
        max_hmag_err = max_hmag_err.max(hmag_err);
        max_rmag_err = max_rmag_err.max(rmag_err);
        max_vmag_err = max_vmag_err.max(vmag_err);

        // Per-step assertions with descriptive messages
        assert!(
            sma_err < 1e-3,
            "t={:.1}s: semi_major_axis error {sma_err:.6e} m exceeds 1e-3 m \
             (ours={:.6}, JEOD={:.6})",
            rec.time, oe.semi_major_axis, rec.semi_major_axis
        );
        assert!(
            ecc_err < 1e-12,
            "t={:.1}s: eccentricity error {ecc_err:.6e} exceeds 1e-12 \
             (ours={:.15e}, JEOD={:.15e})",
            rec.time, oe.e_mag, rec.e_mag
        );
        assert!(
            inc_err < 1e-10,
            "t={:.1}s: inclination error {inc_err:.6e} rad exceeds 1e-10 rad \
             (ours={:.15e}, JEOD={:.15e})",
            rec.time, oe.inclination, rec.inclination
        );
        assert!(
            aop_err < 1e-10,
            "t={:.1}s: arg_periapsis error {aop_err:.6e} rad exceeds 1e-10 rad",
            rec.time
        );
        assert!(
            lan_err < 1e-10,
            "t={:.1}s: long_asc_node error {lan_err:.6e} rad exceeds 1e-10 rad",
            rec.time
        );
        assert!(
            ta_err < 1e-10,
            "t={:.1}s: true_anom error {ta_err:.6e} rad exceeds 1e-10 rad",
            rec.time
        );
        assert!(
            ma_err < 1e-10,
            "t={:.1}s: mean_anom error {ma_err:.6e} rad exceeds 1e-10 rad",
            rec.time
        );

        // Log every 10th record
        if idx % 10 == 0 {
            eprintln!(
                "  t={:>8.1}s: sma_err={:.3e} m, ecc_err={:.3e}, inc_err={:.3e} rad, ta_err={:.3e} rad",
                rec.time, sma_err, ecc_err, inc_err, ta_err
            );
        }
    }

    eprintln!("\n  === Max errors across {} timesteps ===", records.len());
    eprintln!("  semi_major_axis:    {max_sma_err:.6e} m");
    eprintln!("  semiparam:          {max_sp_err:.6e} m");
    eprintln!("  eccentricity:       {max_ecc_err:.6e}");
    eprintln!("  inclination:        {max_inc_err:.6e} rad");
    eprintln!("  arg_periapsis:      {max_aop_err:.6e} rad");
    eprintln!("  long_asc_node:      {max_lan_err:.6e} rad");
    eprintln!("  true_anom:          {max_ta_err:.6e} rad");
    eprintln!("  mean_anom:          {max_ma_err:.6e} rad");
    eprintln!("  mean_motion:        {max_mm_err:.6e} rad/s");
    eprintln!("  orbital_anom:       {max_oa_err:.6e} rad");
    eprintln!("  orb_energy:         {max_energy_err:.6e} J/kg");
    eprintln!("  orb_ang_momentum:   {max_hmag_err:.6e} m^2/s");
    eprintln!("  r_mag:              {max_rmag_err:.6e} m");
    eprintln!("  vel_mag:            {max_vmag_err:.6e} m/s");
}
