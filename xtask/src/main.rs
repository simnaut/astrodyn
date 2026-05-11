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
    publish                 Publish the 13 non-verif crates to crates.io
                            in topological order. Pass --dry-run to
                            walk the sequence without uploading to
                            crates.io (the registry is still queried
                            to resolve dependencies).

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

publish options:
    --dry-run               Run `cargo publish --dry-run` for each
                            crate. No registry changes; no waiting
                            between crates. Useful for validating a
                            republish — for the *first* publish of a
                            crate, dry-run will fail at the second
                            crate (its workspace dep isn't on crates.io
                            yet), so use `cargo package --workspace
                            --no-verify` for first-publish validation.
    --from <crate>          Resume from a specific crate in the
                            sequence. Lets you recover from a
                            mid-sequence failure without re-publishing
                            crates the registry has already accepted.
    --token <token>         Pass-through to `cargo publish --token`.
                            If omitted, cargo reads CARGO_REGISTRY_TOKEN
                            from the environment (the path CI uses).
    --no-wait               Skip the post-publish index-propagation
                            poll. Use only when chasing a transient
                            bug — without the poll, the next crate's
                            publish may fail with `package not found in
                            registry` while the sparse index catches up.
    --allow-dirty           Pass `--allow-dirty` to `cargo publish`.
                            Local-validation escape hatch for running
                            `--dry-run` against an uncommitted working
                            tree; never set this in CI.
    --no-verify             Pass `--no-verify` to `cargo publish`,
                            skipping the build step that resolves
                            dependencies from crates.io. Required for
                            first-publish dry-runs (deps aren't on the
                            registry yet); harmless on republishes.

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
        "publish" => {
            publish(args.collect());
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

// Topological order for `cargo xtask publish`. Each entry must be
// publishable using only registry copies of the entries above it. The
// layers are: (0) astrodyn_quantities; (1) astrodyn_math, _time,
// _ephemeris, _atmosphere; (2) _planet, _frames, _dynamics; (3)
// _gravity, _interactions; (4) astrodyn (root); (5) _bevy, _runner.
const PUBLISH_ORDER: &[&str] = &[
    "astrodyn_quantities",
    "astrodyn_math",
    "astrodyn_time",
    "astrodyn_ephemeris",
    "astrodyn_atmosphere",
    "astrodyn_planet",
    "astrodyn_frames",
    "astrodyn_dynamics",
    "astrodyn_gravity",
    "astrodyn_interactions",
    "astrodyn",
    "astrodyn_bevy",
    "astrodyn_runner",
];

struct PublishArgs {
    dry_run: bool,
    from: Option<String>,
    token: Option<String>,
    no_wait: bool,
    allow_dirty: bool,
    no_verify: bool,
}

impl PublishArgs {
    fn parse(argv: Vec<String>) -> Self {
        let mut a = Self {
            dry_run: false,
            from: None,
            token: None,
            no_wait: false,
            allow_dirty: false,
            no_verify: false,
        };
        let mut iter = argv.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--dry-run" => a.dry_run = true,
                "--no-wait" => a.no_wait = true,
                "--allow-dirty" => a.allow_dirty = true,
                "--no-verify" => a.no_verify = true,
                "--from" => {
                    a.from = Some(iter.next().unwrap_or_else(|| {
                        eprintln!("publish: --from needs a crate name");
                        exit(2);
                    }));
                }
                "--token" => {
                    a.token = Some(iter.next().unwrap_or_else(|| {
                        eprintln!("publish: --token needs a value");
                        exit(2);
                    }));
                }
                "-h" | "--help" => {
                    println!("{HELP}");
                    exit(0);
                }
                other => {
                    eprintln!("publish: unknown arg `{other}`\n\n{HELP}");
                    exit(2);
                }
            }
        }
        a
    }
}

fn publish(argv: Vec<String>) {
    let args = PublishArgs::parse(argv);

    if let Some(ref from) = args.from {
        if !PUBLISH_ORDER.contains(&from.as_str()) {
            eprintln!(
                "publish: --from `{from}` is not in the publish order. \
                 Valid crates: {}",
                PUBLISH_ORDER.join(", ")
            );
            exit(2);
        }
    }

    let start_idx = match args.from.as_deref() {
        Some(name) => PUBLISH_ORDER
            .iter()
            .position(|c| *c == name)
            .expect("validated above"),
        None => 0,
    };

    // All 13 crates ship at the same workspace version. Read it once
    // so `wait_for_index` can poll for the *new version*, not just the
    // crate name (which is already on the registry on every publish
    // after the first).
    let expected_version = workspace_version();

    let total = PUBLISH_ORDER.len();
    for (i, crate_name) in PUBLISH_ORDER.iter().enumerate().skip(start_idx) {
        let step = i + 1;
        eprintln!(
            "publish [{step}/{total}]: {crate_name}{}",
            if args.dry_run { " (dry-run)" } else { "" }
        );

        let mut cmd = Command::new("cargo");
        cmd.arg("publish").arg("-p").arg(crate_name);
        if args.dry_run {
            cmd.arg("--dry-run");
        }
        if args.allow_dirty {
            cmd.arg("--allow-dirty");
        }
        if args.no_verify {
            cmd.arg("--no-verify");
        }
        if let Some(ref token) = args.token {
            cmd.arg("--token").arg(token);
        }

        let status = cmd
            .status()
            .expect("failed to spawn `cargo publish` — is the cargo CLI on PATH?");
        if !status.success() {
            eprintln!(
                "\npublish: `cargo publish -p {crate_name}` failed (step {step}/{total}).\n\
                 To resume after fixing the issue, run:\n\
                 \n    cargo xtask publish --from {crate_name}{}\n",
                if args.dry_run { " --dry-run" } else { "" }
            );
            exit(1);
        }

        // Real publishes need to wait for the crates.io sparse index to
        // pick up the new version before the next crate (which depends
        // on this one via path+version) can resolve it. Dry-runs don't
        // touch the registry, so skip the wait.
        let is_last = i == total - 1;
        if !args.dry_run && !args.no_wait && !is_last {
            wait_for_index(crate_name, &expected_version);
        }
    }

    eprintln!(
        "\npublish: done — {total} crate(s) {}",
        if args.dry_run {
            "dry-run-published"
        } else {
            "published"
        }
    );
}

// Poll `cargo search` until the just-published crate AND VERSION
// appear in the sparse index. `cargo publish` returns success once
// the upload is accepted, but the index can lag by 10–60s before
// path+version deps in the next crate resolve. Matching on the
// version (not just the name) is critical: on every publish after the
// first, the crate name is already on the registry from a prior
// version, so a name-only check would short-circuit while the index
// is still serving stale metadata. Cap at 5 minutes — anything
// longer is almost certainly a registry incident worth a human eye.
fn wait_for_index(crate_name: &str, expected_version: &str) {
    use std::thread::sleep;
    use std::time::{Duration, Instant};

    let deadline = Instant::now() + Duration::from_secs(300);
    let mut attempt = 0u32;
    eprintln!("publish: waiting for crates.io index to serve {crate_name} {expected_version}...");
    // `cargo search foo --limit 1` prints `foo = "x.y.z" # ...` for
    // the latest version. Match on the exact version literal.
    let needle = format!("{crate_name} = \"{expected_version}\"");
    loop {
        attempt += 1;
        let out = Command::new("cargo")
            .args(["search", crate_name, "--limit", "1"])
            .output()
            .expect("failed to spawn `cargo search`");
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if stdout.lines().any(|line| line.starts_with(&needle)) {
                eprintln!("publish: {crate_name} {expected_version} is live on the index.");
                return;
            }
        }
        if Instant::now() >= deadline {
            eprintln!(
                "publish: timed out after 5 min waiting for {crate_name} {expected_version} \
                 on the index. If `cargo publish` succeeded, you can resume with:\n\
                 \n    cargo xtask publish --from <next crate> --no-wait\n"
            );
            exit(1);
        }
        // Backoff: 10s, 10s, 15s, 20s, then 30s. Most publishes land
        // within the first two polls; the longer waits are insurance
        // against a stuck index.
        let delay = match attempt {
            1 | 2 => 10,
            3 => 15,
            4 => 20,
            _ => 30,
        };
        sleep(Duration::from_secs(delay));
    }
}

// Read `[workspace.package].version` from the workspace `Cargo.toml`.
// All 13 publishable crates inherit this value via
// `version.workspace = true`, so we only need to find it once.
fn workspace_version() -> String {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask Cargo.toml has a parent")
        .to_path_buf();
    let manifest_path = workspace_root.join("Cargo.toml");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "failed to read workspace manifest {}: {e}",
            manifest_path.display()
        )
    });
    let mut in_block = false;
    for line in manifest.lines() {
        let trim = line.trim();
        if trim == "[workspace.package]" {
            in_block = true;
            continue;
        }
        if in_block && trim.starts_with('[') {
            break;
        }
        if in_block {
            // Match `version = "X.Y.Z"` (with arbitrary whitespace
            // around the `=`); ignore comments after the value.
            if let Some(rest) = trim.strip_prefix("version") {
                let rest = rest.trim_start();
                if let Some(rest) = rest.strip_prefix('=') {
                    if let Some(v) = rest.trim().split('"').nth(1) {
                        return v.to_string();
                    }
                }
            }
        }
    }
    panic!(
        "could not find `[workspace.package].version` in {}",
        manifest_path.display()
    );
}
