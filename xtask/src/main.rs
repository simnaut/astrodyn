//! `cargo xtask <subcommand>` — workspace dev tooling.
//!
//! Currently the only subcommand is `regenerate-tier3`, a thin wrapper
//! around the Trick / JEOD reference-CSV regeneration Docker invocation.
//! Mission authors and contributors who don't routinely write
//! `docker run -v ... -v ... jeod-trick` invocations can use this to
//! refresh `crates/astrodyn_verif_jeod/test_data/*.csv` from the underlying JEOD verification sims.
//!
//! Background: the regeneration Docker image (`jeod-trick`) is built
//! from `trick/Dockerfile` against the parent directory (so `trick/`
//! and `jeod/` siblings are accessible). It runs `generate_references.sh`
//! inside the container and writes CSVs to a bind-mounted `/output`. The
//! script is itself bind-mounted at runtime (not baked into the image)
//! so iterating on the script doesn't require an image rebuild.
//!
//! See the Tier3-Regeneration wiki page
//! (<https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration>) for the
//! canonical workflow, the incremental-vs-force semantics, and how to add a new sim.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

const HELP: &str = "\
Usage: cargo xtask <subcommand> [args]

Subcommands:
    regenerate-tier3        Regenerate Tier 3 reference CSVs via the
                            jeod-trick Docker image. Incremental by
                            default; pass --force to regenerate all.
    perf-baseline           Run `tier3_perf_runner` under
                            release-with-debug, emit a JSON sample.
                            Wraps the binary so local invocations
                            match the CI `perf-baseline-track` flow.

regenerate-tier3 options:
    --force                 Set FORCE=1 in the container so all sims
                            regenerate even if their CSV already exists.
    --build                 Force rebuild of the jeod-trick image
                            before running.
    --output <path>         Override the host output directory
                            (default: ./crates/astrodyn_verif_jeod/test_data).
    --image <tag>           Override the Docker image tag
                            (default: jeod-trick).
    --max-parallel <n>      Cap concurrent trick-CP builds via
                            MAX_PARALLEL=<n> in the container. Lower if
                            you hit OOM (each build is ~1–2 GB).
                            Default: 4 (script default).

perf-baseline options:
    --scenario <id>         Scenario name (default: earth_moon_clem).
    --steps <N>             Measurement steps per repeat (default: 100000).
    --warmup <N>            Warmup steps per repeat (default: 1000).
    --repeat <K>            Number of independent samples (default: 5).
    --dt <secs>             Integration timestep (default: 0.03125).
    --phase-timing          Build with the `phase_timing` cargo feature
                            and emit per-phase µs/step. Forces a rebuild
                            of `tier3_perf_runner` with the feature on.
    --output <path>         Write JSON to <path> instead of stdout.

    -h, --help              Print this help.
";

fn main() {
    let mut args = env::args().skip(1);
    let Some(subcmd) = args.next() else {
        eprintln!("{HELP}");
        exit(2);
    };

    match subcmd.as_str() {
        "-h" | "--help" | "help" => {
            println!("{HELP}");
        }
        "regenerate-tier3" => {
            regenerate_tier3(args.collect());
        }
        "perf-baseline" => {
            perf_baseline(args.collect());
        }
        other => {
            eprintln!("xtask: unknown subcommand `{other}`\n\n{HELP}");
            exit(2);
        }
    }
}

struct RegenerateArgs {
    force: bool,
    build_image: bool,
    output: PathBuf,
    image: String,
    max_parallel: Option<u32>,
}

impl RegenerateArgs {
    fn parse(argv: Vec<String>) -> Self {
        let mut a = Self {
            force: false,
            build_image: false,
            output: PathBuf::from("crates/astrodyn_verif_jeod/test_data"),
            image: "jeod-trick".to_string(),
            max_parallel: None,
        };
        let mut iter = argv.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--force" => a.force = true,
                "--build" => a.build_image = true,
                "--output" => {
                    a.output = PathBuf::from(iter.next().unwrap_or_else(|| {
                        eprintln!("regenerate-tier3: --output needs a value");
                        exit(2);
                    }));
                }
                "--image" => {
                    a.image = iter.next().unwrap_or_else(|| {
                        eprintln!("regenerate-tier3: --image needs a value");
                        exit(2);
                    });
                }
                "--max-parallel" => {
                    let raw = iter.next().unwrap_or_else(|| {
                        eprintln!("regenerate-tier3: --max-parallel needs a value");
                        exit(2);
                    });
                    let n: u32 = raw.parse().unwrap_or_else(|e| {
                        eprintln!("regenerate-tier3: --max-parallel `{raw}` is not a positive integer: {e}");
                        exit(2);
                    });
                    // Reject 0 — `generate_references.sh`'s `throttled_bg`
                    // uses `while [ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]`
                    // which spins forever when MAX_PARALLEL=0 and the PID
                    // list is empty. Match the error message to the actual
                    // validation.
                    if n == 0 {
                        eprintln!(
                            "regenerate-tier3: --max-parallel must be ≥ 1 (got 0; \
                             0 would deadlock the throttling loop in generate_references.sh)"
                        );
                        exit(2);
                    }
                    a.max_parallel = Some(n);
                }
                "-h" | "--help" => {
                    println!("{HELP}");
                    exit(0);
                }
                other => {
                    eprintln!("regenerate-tier3: unknown arg `{other}`\n\n{HELP}");
                    exit(2);
                }
            }
        }
        a
    }
}

fn regenerate_tier3(argv: Vec<String>) {
    let args = RegenerateArgs::parse(argv);

    // Workspace root = directory containing the workspace Cargo.toml.
    // CARGO_MANIFEST_DIR points at xtask/, so go up one level.
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask Cargo.toml has a parent")
        .to_path_buf();

    let output_abs = if args.output.is_absolute() {
        args.output.clone()
    } else {
        workspace_root.join(&args.output)
    };
    std::fs::create_dir_all(&output_abs).unwrap_or_else(|e| {
        eprintln!(
            "regenerate-tier3: cannot create output dir {}: {e}",
            output_abs.display()
        );
        exit(1);
    });

    let dockerfile = workspace_root.join("trick/Dockerfile");
    let script = workspace_root.join("trick/generate_references.sh");
    assert_paths_exist(&[&dockerfile, &script]);

    if args.build_image {
        run_build(&workspace_root, &dockerfile, &args.image);
    } else if !image_exists(&args.image) {
        eprintln!(
            "regenerate-tier3: image `{}` not found; building it now (use --build to force rebuild)",
            args.image
        );
        run_build(&workspace_root, &dockerfile, &args.image);
    }

    run_regenerate(
        &output_abs,
        &script,
        &args.image,
        args.force,
        args.max_parallel,
    );
}

fn assert_paths_exist(paths: &[&Path]) {
    for p in paths {
        if !p.exists() {
            eprintln!("regenerate-tier3: required path missing: {}", p.display());
            exit(1);
        }
    }
}

fn image_exists(tag: &str) -> bool {
    let out = Command::new("docker")
        .args(["image", "inspect", tag])
        .output();
    matches!(out, Ok(o) if o.status.success())
}

fn run_build(workspace_root: &Path, dockerfile: &Path, tag: &str) {
    // The Docker build context is the *parent* of the workspace root,
    // so that both `trick/` and `jeod/` siblings can be COPYed into the
    // image. This matches the convention in the Dockerfile header.
    let build_context = workspace_root
        .parent()
        .expect("workspace root has a parent");
    // The Dockerfile has `ARG BEVY_JEOD_DIR=astrodyn` and uses it to
    // COPY the entrypoint and other workspace files. Derive the actual
    // directory name from `workspace_root` so checkouts in
    // differently-named directories (e.g. `astrodyn_fork`, `my_clone`,
    // worktrees with autogenerated names) build correctly without manual
    // intervention.
    let workspace_dir_name = workspace_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("astrodyn");
    eprintln!(
        "regenerate-tier3: building image `{tag}` from {} (context: {}, BEVY_JEOD_DIR={})",
        dockerfile.display(),
        build_context.display(),
        workspace_dir_name,
    );
    let status = Command::new("docker")
        .arg("build")
        .arg("-f")
        .arg(dockerfile)
        .arg("--build-arg")
        .arg(format!("BEVY_JEOD_DIR={workspace_dir_name}"))
        .arg("-t")
        .arg(tag)
        .arg(build_context)
        .status()
        .expect("failed to spawn `docker build` — is the docker CLI installed and on PATH?");
    if !status.success() {
        eprintln!("regenerate-tier3: docker build failed");
        exit(1);
    }
}

fn run_regenerate(output: &Path, script: &Path, tag: &str, force: bool, max_parallel: Option<u32>) {
    let mut cmd = Command::new("docker");
    cmd.arg("run").arg("--rm");
    if force {
        cmd.args(["-e", "FORCE=1"]);
    }
    if let Some(n) = max_parallel {
        cmd.arg("-e").arg(format!("MAX_PARALLEL={n}"));
    }
    cmd.arg("-v").arg(format!("{}:/output", output.display()));
    cmd.arg("-v")
        .arg(format!("{}:/generate_references.sh:ro", script.display()));
    cmd.arg(tag);

    eprintln!(
        "regenerate-tier3: regenerating into {} (force={force}, max_parallel={})",
        output.display(),
        max_parallel
            .map(|n| n.to_string())
            .unwrap_or_else(|| "default".into())
    );
    let status = cmd
        .status()
        .expect("failed to spawn `docker run` — is the docker CLI installed and on PATH?");
    if !status.success() {
        eprintln!("regenerate-tier3: docker run failed");
        exit(1);
    }
    eprintln!("regenerate-tier3: done.");
}

struct PerfBaselineArgs {
    scenario: String,
    steps: usize,
    warmup: usize,
    repeat: usize,
    dt: f64,
    phase_timing: bool,
    output: Option<PathBuf>,
}

impl PerfBaselineArgs {
    fn parse(argv: Vec<String>) -> Self {
        let mut a = Self {
            scenario: "earth_moon_clem".to_string(),
            steps: 100_000,
            warmup: 1_000,
            repeat: 5,
            dt: 0.03125,
            phase_timing: false,
            output: None,
        };
        let mut iter = argv.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--scenario" => {
                    a.scenario = iter.next().unwrap_or_else(|| {
                        eprintln!("perf-baseline: --scenario needs a value");
                        exit(2);
                    });
                }
                "--steps" => {
                    let v = iter.next().unwrap_or_else(|| {
                        eprintln!("perf-baseline: --steps needs a value");
                        exit(2);
                    });
                    a.steps = v.parse().unwrap_or_else(|e| {
                        eprintln!("perf-baseline: --steps `{v}` is not a usize: {e}");
                        exit(2);
                    });
                }
                "--warmup" => {
                    let v = iter.next().unwrap_or_else(|| {
                        eprintln!("perf-baseline: --warmup needs a value");
                        exit(2);
                    });
                    a.warmup = v.parse().unwrap_or_else(|e| {
                        eprintln!("perf-baseline: --warmup `{v}` is not a usize: {e}");
                        exit(2);
                    });
                }
                "--repeat" => {
                    let v = iter.next().unwrap_or_else(|| {
                        eprintln!("perf-baseline: --repeat needs a value");
                        exit(2);
                    });
                    a.repeat = v.parse().unwrap_or_else(|e| {
                        eprintln!("perf-baseline: --repeat `{v}` is not a usize: {e}");
                        exit(2);
                    });
                }
                "--dt" => {
                    let v = iter.next().unwrap_or_else(|| {
                        eprintln!("perf-baseline: --dt needs a value");
                        exit(2);
                    });
                    a.dt = v.parse().unwrap_or_else(|e| {
                        eprintln!("perf-baseline: --dt `{v}` is not a f64: {e}");
                        exit(2);
                    });
                }
                "--phase-timing" => a.phase_timing = true,
                "--output" => {
                    a.output = Some(PathBuf::from(iter.next().unwrap_or_else(|| {
                        eprintln!("perf-baseline: --output needs a value");
                        exit(2);
                    })));
                }
                "-h" | "--help" => {
                    println!("{HELP}");
                    exit(0);
                }
                other => {
                    eprintln!("perf-baseline: unknown arg `{other}`\n\n{HELP}");
                    exit(2);
                }
            }
        }
        a
    }
}

fn perf_baseline(argv: Vec<String>) {
    let args = PerfBaselineArgs::parse(argv);

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask Cargo.toml has a parent")
        .to_path_buf();

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&workspace_root);
    cmd.args([
        "run",
        "--profile",
        "release-with-debug",
        "-p",
        "astrodyn_verif_jeod",
        "--bin",
        "tier3_perf_runner",
    ]);
    if args.phase_timing {
        cmd.args(["--features", "astrodyn_verif_jeod/phase_timing"]);
    }
    cmd.arg("--");
    cmd.args(["--scenario", &args.scenario]);
    cmd.args(["--steps", &args.steps.to_string()]);
    cmd.args(["--warmup", &args.warmup.to_string()]);
    cmd.args(["--repeat", &args.repeat.to_string()]);
    cmd.args(["--dt", &args.dt.to_string()]);
    if args.phase_timing {
        cmd.arg("--phase-timing");
    }
    if let Some(path) = &args.output {
        cmd.arg("--output").arg(path);
    }

    eprintln!("perf-baseline: invoking {cmd:?}");
    let status = cmd
        .status()
        .expect("failed to spawn `cargo run` — is cargo on PATH?");
    if !status.success() {
        eprintln!("perf-baseline: tier3_perf_runner exited non-zero");
        exit(status.code().unwrap_or(1));
    }
}
