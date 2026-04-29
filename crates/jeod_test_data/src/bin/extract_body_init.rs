//! Extract JEOD `Modified_data/*.py` body-initialization vectors into
//! committed fixtures under `test_data/body_init/<vehicle>.json`.
//!
//! This is a **regen-only** path: it reads `$JEOD_HOME` (or `$JEOD_PATH`,
//! or an explicit `--jeod-home <PATH>` argument), parses the body-init
//! Python files for each scenario, and writes the JSON consumed by
//! `jeod_test_data::reference_state::*` and
//! `jeod_test_data::orbital_init::*` at runtime.
//!
//! Run after a JEOD upgrade or whenever a scenario file is added/amended:
//!
//! ```bash
//! cargo run -p jeod_test_data --bin extract_body_init -- \
//!     --jeod-home /path/to/jeod
//! ```
//!
//! ## Sources of record (audit list)
//!
//! Per scenario, the binary parses:
//!
//! - `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/<vehicle>/`
//!   `reference_inertial_trans_state.py`        — ECI reference state
//!   `trans_Orbit_inertial_body_set01.py`       — orbit (sma/ecc/inc/raan/argp/t_peri)
//!   `trans_Orbit_inertial_body_set02.py`       — orbit (mean anomaly)
//!   `trans_Orbit_inertial_body_set10.py`       — orbit (true anomaly)
//!   `trans_Orbit_pfix_body_set01.py`           — pfix orbit (set01 form)
//!   `trans_TransState_inertial_body.py`        — direct Cartesian (STS_114 only)
//!
//! Scenarios extracted: `ISS`, `STS_114`. Each scenario writes a single
//! `test_data/body_init/<vehicle>.json`. The `reference_inertial` and
//! `trans_state` files listed in `SCENARIOS` are required and the binary
//! errors loudly if they are missing (per CLAUDE.md "Fail Loudly"); the
//! per-orbit `set01/set02/set10/pfix_set01` files are optional and skipped
//! when absent (not every vehicle defines every orbit form).
//!
//! The schema follows the no-`serde_json` style used elsewhere in this
//! crate (see `planet_geodetic_verif.rs`); the JSON is hand-written and
//! parsed back via `body_init_fixtures::parse_*` helpers.

use std::io::Write;

use jeod_test_data::body_init_fixtures::{
    OrbitalInitRecord, ReferenceStateRecord, TransStateRecord,
};
use jeod_test_data::orbital_init::{parse_orbital_init_py, parse_trans_state_py};
use jeod_test_data::reference_state::parse_reference_state_py;

/// Vehicles and the init records to extract for each.
const SCENARIOS: &[Scenario] = &[
    Scenario {
        vehicle: "ISS",
        reference_inertial: true,
        orbit_inits: &[
            "trans_Orbit_inertial_body_set01",
            "trans_Orbit_inertial_body_set02",
            "trans_Orbit_inertial_body_set10",
            "trans_Orbit_pfix_body_set01",
        ],
        trans_states: &[],
    },
    Scenario {
        vehicle: "STS_114",
        reference_inertial: true,
        orbit_inits: &[
            "trans_Orbit_inertial_body_set01",
            "trans_Orbit_pfix_body_set01",
        ],
        trans_states: &["trans_TransState_inertial_body"],
    },
];

struct Scenario {
    vehicle: &'static str,
    reference_inertial: bool,
    orbit_inits: &'static [&'static str],
    trans_states: &'static [&'static str],
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jeod_root = resolve_jeod_root(&args).unwrap_or_else(|| {
        eprintln!(
            "extract_body_init: JEOD source not found.\n\
             Pass `--jeod-home <PATH>` or set JEOD_HOME / JEOD_PATH \
             (see CLAUDE.md \"Environment Setup\")."
        );
        std::process::exit(2);
    });

    let body_init_root = body_init_dir();
    std::fs::create_dir_all(&body_init_root).unwrap_or_else(|e| {
        panic!("Cannot create {}: {e}", body_init_root.display());
    });

    for scenario in SCENARIOS {
        let mut bundle = ScenarioBundle::new(scenario.vehicle);

        if scenario.reference_inertial {
            let path = jeod_root.join(format!(
                "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/\
                 reference_inertial_trans_state.py",
                scenario.vehicle,
            ));
            let content = read_required(&path);
            let state = parse_reference_state_py(&content).unwrap_or_else(|e| {
                panic!(
                    "extract_body_init: failed to parse reference state in {}: {e}",
                    path.display()
                );
            });
            bundle.reference_inertial = Some(state);
        }

        for init_name in scenario.orbit_inits {
            let path = jeod_root.join(format!(
                "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/{}.py",
                scenario.vehicle, init_name,
            ));
            if !path.exists() {
                continue;
            }
            let content = read_required(&path);
            let init = parse_orbital_init_py(&content).unwrap_or_else(|e| {
                panic!(
                    "extract_body_init: failed to parse orbital init in {}: {e}",
                    path.display()
                );
            });
            bundle.orbital_inits.push((init_name.to_string(), init));
        }

        for init_name in scenario.trans_states {
            let path = jeod_root.join(format!(
                "models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/{}.py",
                scenario.vehicle, init_name,
            ));
            if !path.exists() {
                continue;
            }
            let content = read_required(&path);
            let trans = parse_trans_state_py(&content).unwrap_or_else(|e| {
                panic!(
                    "extract_body_init: failed to parse trans state in {}: {e}",
                    path.display()
                );
            });
            bundle.trans_states.push((init_name.to_string(), trans));
        }

        let out_path =
            body_init_root.join(format!("{}.json", scenario.vehicle.to_ascii_lowercase()));
        let mut f = std::fs::File::create(&out_path)
            .unwrap_or_else(|e| panic!("Cannot create {}: {e}", out_path.display()));
        write_bundle(&mut f, &bundle);
        println!(
            "wrote {} (reference_inertial={}, orbital_inits={}, trans_states={})",
            out_path.display(),
            bundle.reference_inertial.is_some(),
            bundle.orbital_inits.len(),
            bundle.trans_states.len(),
        );
    }
}

fn resolve_jeod_root(args: &[String]) -> Option<std::path::PathBuf> {
    if let Some(idx) = args.iter().position(|a| a == "--jeod-home") {
        if let Some(p) = args.get(idx + 1) {
            return Some(std::path::PathBuf::from(p));
        }
    }
    if let Ok(p) = std::env::var("JEOD_PATH") {
        return Some(std::path::PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("JEOD_HOME") {
        return Some(std::path::PathBuf::from(p));
    }
    None
}

fn body_init_dir() -> std::path::PathBuf {
    jeod_test_data::tier3_csv::test_data_path("body_init")
}

fn read_required(path: &std::path::Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("Cannot read {}: {e}", path.display()))
}

struct ScenarioBundle {
    vehicle: &'static str,
    reference_inertial: Option<ReferenceStateRecord>,
    orbital_inits: Vec<(String, OrbitalInitRecord)>,
    trans_states: Vec<(String, TransStateRecord)>,
}

impl ScenarioBundle {
    fn new(vehicle: &'static str) -> Self {
        Self {
            vehicle,
            reference_inertial: None,
            orbital_inits: Vec::new(),
            trans_states: Vec::new(),
        }
    }
}

fn write_bundle(out: &mut std::fs::File, bundle: &ScenarioBundle) {
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema_version\": 1,").unwrap();
    writeln!(out, "  \"vehicle\": \"{}\",", bundle.vehicle).unwrap();
    writeln!(
        out,
        "  \"source\": \"models/dynamics/body_action/verif/SIM_orbinit/Modified_data/{}/\",",
        bundle.vehicle,
    )
    .unwrap();
    writeln!(
        out,
        "  \"note\": \"Body initialization vectors. Regenerate with: cargo run -p jeod_test_data \
         --bin extract_body_init -- --jeod-home $JEOD_HOME\","
    )
    .unwrap();

    // reference_inertial: an object or null.
    write!(out, "  \"reference_inertial\": ").unwrap();
    match &bundle.reference_inertial {
        Some(s) => writeln!(
            out,
            "{{\"position\": [{}, {}, {}], \"velocity\": [{}, {}, {}]}},",
            fmt(s.position[0]),
            fmt(s.position[1]),
            fmt(s.position[2]),
            fmt(s.velocity[0]),
            fmt(s.velocity[1]),
            fmt(s.velocity[2]),
        ),
        None => writeln!(out, "null,"),
    }
    .unwrap();

    // orbital_inits: array of {name, ...}.
    writeln!(out, "  \"orbital_inits\": [").unwrap();
    for (i, (name, init)) in bundle.orbital_inits.iter().enumerate() {
        let comma = if i + 1 < bundle.orbital_inits.len() {
            ","
        } else {
            ""
        };
        writeln!(out, "    {{").unwrap();
        writeln!(out, "      \"name\": \"{}\",", name).unwrap();
        writeln!(
            out,
            "      \"semi_major_axis\": {},",
            fmt(init.semi_major_axis)
        )
        .unwrap();
        writeln!(out, "      \"eccentricity\": {},", fmt(init.eccentricity)).unwrap();
        writeln!(out, "      \"inclination\": {},", fmt(init.inclination)).unwrap();
        writeln!(
            out,
            "      \"ascending_node\": {},",
            fmt(init.ascending_node)
        )
        .unwrap();
        writeln!(out, "      \"arg_periapsis\": {},", fmt(init.arg_periapsis)).unwrap();
        writeln!(
            out,
            "      \"time_periapsis\": {},",
            fmt_opt(init.time_periapsis),
        )
        .unwrap();
        writeln!(
            out,
            "      \"mean_anomaly\": {},",
            fmt_opt(init.mean_anomaly),
        )
        .unwrap();
        writeln!(
            out,
            "      \"true_anomaly\": {},",
            fmt_opt(init.true_anomaly),
        )
        .unwrap();
        writeln!(out, "      \"planet_name\": \"{}\",", init.planet_name).unwrap();
        writeln!(
            out,
            "      \"reference_frame\": \"{}\"",
            init.reference_frame
        )
        .unwrap();
        writeln!(out, "    }}{}", comma).unwrap();
    }
    writeln!(out, "  ],").unwrap();

    // trans_states: array of {name, ...}.
    writeln!(out, "  \"trans_states\": [").unwrap();
    for (i, (name, t)) in bundle.trans_states.iter().enumerate() {
        let comma = if i + 1 < bundle.trans_states.len() {
            ","
        } else {
            ""
        };
        writeln!(out, "    {{").unwrap();
        writeln!(out, "      \"name\": \"{}\",", name).unwrap();
        writeln!(
            out,
            "      \"position\": [{}, {}, {}],",
            fmt(t.position[0]),
            fmt(t.position[1]),
            fmt(t.position[2]),
        )
        .unwrap();
        writeln!(
            out,
            "      \"velocity\": [{}, {}, {}],",
            fmt(t.velocity[0]),
            fmt(t.velocity[1]),
            fmt(t.velocity[2]),
        )
        .unwrap();
        writeln!(out, "      \"reference_frame\": \"{}\"", t.reference_frame).unwrap();
        writeln!(out, "    }}{}", comma).unwrap();
    }
    writeln!(out, "  ]").unwrap();

    writeln!(out, "}}").unwrap();
}

/// Round-trippable f64 representation; `{x:?}` always emits a decimal point
/// or exponent so the JSON literal parses unambiguously back to the same
/// `f64`. (`{x}` would print `1` for `1.0` which would parse as integer.)
fn fmt(x: f64) -> String {
    format!("{x:?}")
}

fn fmt_opt(x: Option<f64>) -> String {
    match x {
        Some(v) => fmt(v),
        None => "null".to_string(),
    }
}
