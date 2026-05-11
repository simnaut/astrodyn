//! Steady-state per-step wall-clock measurement for Tier 3 scenarios.
//!
//! Drives a named scenario through `astrodyn_runner::Simulation::step()`
//! many times, captures elapsed time across `--repeat` independent runs,
//! and emits a single JSON object with mean / stdev / per-step µs / peak
//! RSS. Used by the `xtask perf-baseline` wrapper and CI's
//! `perf-baseline-track` job. Designed for stability across runs — one
//! cold setup per repeat (catches setup-time regressions), explicit
//! warmup window (absorbs first-step transients), no shared mutable
//! state across repeats.
//!
//! ## CLI
//!
//! ```text
//! tier3_perf_runner --scenario <id> [--steps N] [--warmup N] [--repeat K]
//!                   [--dt SECS] [--phase-timing] [--output PATH]
//! ```
//!
//! `--scenario` is required. Known scenarios: `earth_moon_clem`. Future
//! scenarios are added by extending [`build_scenario`].
//!
//! `--phase-timing` is gated by the `phase_timing` cargo feature; the
//! binary panics with a rebuild command if the flag is passed without
//! the feature. The `xtask perf-baseline` wrapper handles the feature
//! flag automatically.
//!
//! ## Output schema
//!
//! Single JSON object on stdout (or `--output` path). Issue #447's
//! schema; see the README's Performance toolkit section for the
//! consumer (`perf-history.csv` append in `perf-baseline-track`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use astrodyn::SimulationBuilder;
use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::setups;

const HELP: &str = "\
tier3_perf_runner — steady-state per-step performance measurement.

Usage:
    tier3_perf_runner --scenario <id> [options]

Options:
    --scenario <id>     Required. Known: earth_moon_clem
    --steps <N>         Measurement steps per repeat (default: 100000)
    --warmup <N>        Warmup steps per repeat (default: 1000)
    --repeat <K>        Number of independent samples (default: 5)
    --dt <secs>         Integration timestep (default: 0.03125)
    --phase-timing      Emit per-phase µs/step. Requires the
                        `phase_timing` cargo feature.
    --output <path>     Write JSON to <path> instead of stdout
    -h, --help          Print this help
";

struct PerfArgs {
    scenario: String,
    steps: usize,
    warmup: usize,
    repeat: usize,
    dt: f64,
    phase_timing: bool,
    output: Option<PathBuf>,
}

impl PerfArgs {
    fn parse(argv: Vec<String>) -> Result<Self, String> {
        let mut a = Self {
            scenario: String::new(),
            steps: 100_000,
            warmup: 1_000,
            repeat: 5,
            dt: 0.03125,
            phase_timing: false,
            output: None,
        };
        let mut it = argv.into_iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    println!("{HELP}");
                    std::process::exit(0);
                }
                "--scenario" => {
                    a.scenario = it
                        .next()
                        .ok_or_else(|| "--scenario requires a value".to_string())?;
                }
                "--steps" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--steps requires a value".to_string())?;
                    a.steps = v.parse().map_err(|e| format!("--steps {v:?}: {e}"))?;
                }
                "--warmup" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--warmup requires a value".to_string())?;
                    a.warmup = v.parse().map_err(|e| format!("--warmup {v:?}: {e}"))?;
                }
                "--repeat" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--repeat requires a value".to_string())?;
                    a.repeat = v.parse().map_err(|e| format!("--repeat {v:?}: {e}"))?;
                }
                "--dt" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--dt requires a value".to_string())?;
                    a.dt = v.parse().map_err(|e| format!("--dt {v:?}: {e}"))?;
                }
                "--phase-timing" => {
                    a.phase_timing = true;
                }
                "--output" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--output requires a value".to_string())?;
                    a.output = Some(PathBuf::from(v));
                }
                other => {
                    return Err(format!("unknown argument {other:?}; --help for usage"));
                }
            }
        }
        if a.scenario.is_empty() {
            return Err("--scenario is required".into());
        }
        if a.repeat == 0 {
            return Err("--repeat must be >= 1".into());
        }
        if a.steps == 0 {
            return Err("--steps must be >= 1".into());
        }
        Ok(a)
    }
}

/// Resolve a scenario name to a fully-wired [`SimulationBuilder`].
///
/// Seeded with `earth_moon_clem`; siblings land alongside as the perf
/// matrix expands. Unknown names panic — fail loudly per the
/// non-negotiable invariant.
fn build_scenario(name: &str, dt: f64) -> SimulationBuilder {
    match name {
        "earth_moon_clem" => setups::earth_moon_clem::earth_moon_clem(dt, None),
        other => panic!(
            "unknown scenario {other:?} (known: earth_moon_clem). \
             Add a new arm to `tier3_perf_runner::build_scenario` and a \
             setup function under `astrodyn_verif_jeod::setups`."
        ),
    }
}

/// Linux peak resident set size (VmHWM) in megabytes. Returns 0.0 on
/// non-Linux hosts (the field is encoded as 0.0 in the JSON and the
/// operator interprets the value as Linux-only). CI is Linux; that's
/// the supported path.
fn read_vm_hwm_mb() -> f64 {
    let content = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

/// Short git SHA at build time. Reads `git rev-parse --short HEAD`
/// lazily; empty string on failure (running outside a git checkout).
fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

/// Cargo profile this binary was built under. Static — read at compile
/// time from `cfg(debug_assertions)`.
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "dev"
    } else {
        // No way to distinguish `release` from `release-with-debug` at
        // runtime without a build script; the wrapper sets the right
        // profile and the field is informational.
        "release"
    }
}

/// Compile-time rustc version string.
fn rustc_version() -> String {
    // Use rustversion crate? No — adds a build dep. Empty is fine; the
    // perf-history consumer can correlate via git_sha.
    env!("CARGO_PKG_RUST_VERSION").to_string()
}

/// Result of one repeat: wall-clock window + accumulated phase
/// timings (under `phase_timing`) + peak RSS sampled after the run.
struct Sample {
    elapsed: Duration,
    #[cfg(feature = "phase_timing")]
    timings: astrodyn_runner::PhaseTimings,
    rss_mb: f64,
}

#[cfg(feature = "phase_timing")]
fn run_one_repeat(args: &PerfArgs) -> Sample {
    let mut sim = build_scenario(&args.scenario, args.dt)
        .build()
        .expect("scenario must validate");
    sim.step_n(args.warmup).expect("warmup");
    sim.reset_phase_timings();
    let t0 = Instant::now();
    sim.step_n(args.steps).expect("measurement");
    let elapsed = t0.elapsed();
    Sample {
        elapsed,
        timings: *sim.phase_timings(),
        rss_mb: read_vm_hwm_mb(),
    }
}

#[cfg(not(feature = "phase_timing"))]
fn run_one_repeat(args: &PerfArgs) -> Sample {
    let mut sim = build_scenario(&args.scenario, args.dt)
        .build()
        .expect("scenario must validate");
    sim.step_n(args.warmup).expect("warmup");
    let t0 = Instant::now();
    sim.step_n(args.steps).expect("measurement");
    let elapsed = t0.elapsed();
    Sample {
        elapsed,
        rss_mb: read_vm_hwm_mb(),
    }
}

/// Format a f64 with a fixed number of decimals into a JSON-safe
/// number literal (no trailing `.0` for integer values is fine —
/// every consumer is `jq`).
fn fmt_f64(x: f64, decimals: usize) -> String {
    format!("{x:.*}", decimals)
}

/// JSON-escape a string literal. Backslash, double-quote, and control
/// characters; everything else is passed through.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Build the JSON output string from the run results.
fn emit_json(args: &PerfArgs, samples: &[Sample]) -> String {
    let n = samples.len() as f64;
    let secs: Vec<f64> = samples.iter().map(|s| s.elapsed.as_secs_f64()).collect();
    let mean = secs.iter().sum::<f64>() / n;
    let var = if n > 1.0 {
        secs.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let stdev = var.sqrt();
    let per_step_us = if args.steps > 0 {
        1e6 * mean / args.steps as f64
    } else {
        0.0
    };
    let max_rss_mb = samples.iter().map(|s| s.rss_mb).fold(0.0_f64, f64::max);

    let samples_json = secs
        .iter()
        .map(|s| fmt_f64(*s, 6))
        .collect::<Vec<_>>()
        .join(", ");

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"scenario\": \"{}\",\n",
        json_escape(&args.scenario)
    ));
    out.push_str(&format!("  \"dt\": {},\n", fmt_f64(args.dt, 8)));
    out.push_str(&format!("  \"steps\": {},\n", args.steps));
    out.push_str(&format!("  \"warmup\": {},\n", args.warmup));
    out.push_str(&format!("  \"repeat\": {},\n", args.repeat));
    out.push_str(&format!("  \"mean_secs\": {},\n", fmt_f64(mean, 6)));
    out.push_str(&format!("  \"stdev_secs\": {},\n", fmt_f64(stdev, 6)));
    out.push_str(&format!(
        "  \"per_step_us\": {},\n",
        fmt_f64(per_step_us, 4)
    ));
    out.push_str(&format!("  \"max_rss_mb\": {},\n", fmt_f64(max_rss_mb, 2)));
    out.push_str(&format!("  \"samples_secs\": [{}],\n", samples_json));

    #[cfg(feature = "phase_timing")]
    if args.phase_timing {
        emit_phase_timings(&mut out, args, samples);
    }
    let _ = args; // suppress unused warning when feature is off

    out.push_str("  \"host\": {\n");
    out.push_str(&format!(
        "    \"os\": \"{}\",\n",
        json_escape(std::env::consts::OS)
    ));
    out.push_str(&format!(
        "    \"arch\": \"{}\",\n",
        json_escape(std::env::consts::ARCH)
    ));
    out.push_str(&format!(
        "    \"rustc\": \"{}\",\n",
        json_escape(&rustc_version())
    ));
    out.push_str(&format!(
        "    \"git_sha\": \"{}\",\n",
        json_escape(&git_sha())
    ));
    out.push_str(&format!(
        "    \"profile\": \"{}\"\n",
        json_escape(build_profile())
    ));
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

#[cfg(feature = "phase_timing")]
fn emit_phase_timings(out: &mut String, args: &PerfArgs, samples: &[Sample]) {
    let total_steps: u64 = samples.iter().map(|s| s.timings.steps).sum();
    let denom = total_steps.max(1) as f64;
    let sum_us = |getter: fn(&astrodyn_runner::PhaseTimings) -> std::time::Duration| -> f64 {
        let total: std::time::Duration = samples.iter().map(|s| getter(&s.timings)).sum();
        1e6 * total.as_secs_f64() / denom
    };
    out.push_str("  \"phase_timings\": {\n");
    out.push_str(&format!("    \"steps\": {},\n", total_steps));
    let _ = args;
    let phases: &[(
        &str,
        fn(&astrodyn_runner::PhaseTimings) -> std::time::Duration,
    )] = &[
        ("time_advance", |t| t.time_advance),
        ("ephemeris", |t| t.ephemeris),
        ("mass_recompute", |t| t.mass_recompute),
        ("integ_origins_pre", |t| t.integ_origins_pre),
        ("kinematic_pre", |t| t.kinematic_pre),
        ("environment", |t| t.environment),
        ("interactions", |t| t.interactions),
        ("integration", |t| t.integration),
        ("integ_origins_post", |t| t.integ_origins_post),
        ("kinematic_post", |t| t.kinematic_post),
        ("derived", |t| t.derived),
        ("detached_subtrees", |t| t.detached_subtrees),
    ];
    for (i, (name, getter)) in phases.iter().enumerate() {
        let sep = if i + 1 == phases.len() { "" } else { "," };
        out.push_str(&format!(
            "    \"{}_us_per_step\": {}{}\n",
            name,
            fmt_f64(sum_us(*getter), 4),
            sep,
        ));
    }
    out.push_str("  },\n");
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match PerfArgs::parse(argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("tier3_perf_runner: {e}");
            return ExitCode::from(2);
        }
    };

    // Runtime guard for the cargo-feature gate. The xtask wrapper
    // injects `--features astrodyn_verif_jeod/phase_timing` when
    // `--phase-timing` is requested; direct invocations need to do the
    // same. Fail loudly with the actionable rebuild command.
    if args.phase_timing && cfg!(not(feature = "phase_timing")) {
        eprintln!(
            "tier3_perf_runner: --phase-timing requires the `phase_timing` cargo feature.\n\
             Rebuild with:\n  \
             cargo run --profile release-with-debug -p astrodyn_verif_jeod \\\n    \
             --features astrodyn_verif_jeod/phase_timing --bin tier3_perf_runner -- \\\n    \
             <your args>"
        );
        return ExitCode::from(2);
    }

    let mut samples: Vec<Sample> = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        samples.push(run_one_repeat(&args));
    }

    let json = emit_json(&args, &samples);
    match &args.output {
        Some(path) => std::fs::write(path, &json).expect("write --output"),
        None => print!("{json}"),
    }
    ExitCode::from(0)
}
