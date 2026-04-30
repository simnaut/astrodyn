//! Verbatim JEOD source-file fixtures committed under
//! `test_data/jeod_inputs/<jeod-relative-path>`.
//!
//! Tier 3 verification rigs need JEOD configuration inputs (S_define
//! `#define DYNAMICS` lines, `Modified_data/*.py`, `SET_test/RUN_*/input.py`)
//! to recover sim-specific dt, mass properties, time-scale offsets, and
//! gravity-control parameters. Pre #249 those were read live from
//! `$JEOD_HOME`; now the same files are committed verbatim so
//! `cargo nextest run --workspace` works on a fresh clone with no JEOD
//! checkout.
//!
//! ## Layout
//!
//! Every fixture is committed at the same relative path it occupies in
//! a JEOD checkout. For example,
//! `<jeod>/verif/SIM_dyncomp/Modified_data/mass.py` is committed at
//! `test_data/jeod_inputs/verif/SIM_dyncomp/Modified_data/mass.py`.
//! The mirror keeps the audit trail trivial — `diff` against any JEOD
//! checkout to confirm we have the upstream contents.
//!
//! ## Regenerate after a JEOD upgrade
//!
//! See `test_data/jeod_inputs/README.md` for the `cp` recipe. New
//! fixtures are added by copying from `$JEOD_HOME` into the mirror
//! directory and committing the result.

use std::path::{Path, PathBuf};

/// Resolve a JEOD-relative path against the committed
/// `test_data/jeod_inputs/` mirror.
///
/// `relative` is the JEOD-source-relative path
/// (e.g. `"verif/SIM_dyncomp/S_define"` or
/// `"models/dynamics/derived_state/verif/SIM_NED/Modified_data/date_and_time.py"`).
/// Panics with a fail-loudly diagnostic if the resolved path does not
/// exist — committed fixtures must always be present.
pub fn path(relative: &str) -> PathBuf {
    let p = workspace_root()
        .join("test_data/jeod_inputs")
        .join(relative);
    assert!(
        p.exists(),
        "JEOD input fixture not found at {}. \
         If you added a new sim, copy the file from $JEOD_HOME/{relative} \
         into test_data/jeod_inputs/{relative} and commit it. \
         See the module-level docs and test_data/jeod_inputs/README.md.",
        p.display(),
    );
    p
}

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    // Fallback: two parents up from `crates/jeod_test_data/`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .expect("workspace root: CARGO_MANIFEST_DIR has at least two ancestors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_committed_dyncomp_s_define() {
        // The SIM_dyncomp S_define is committed and consumed by every
        // Tier 3 dyncomp test. If this assertion fails the mirror is
        // broken; do not paper over it by skipping the test.
        let p = path("verif/SIM_dyncomp/S_define");
        assert!(
            p.is_file(),
            "expected committed S_define at {}",
            p.display()
        );
    }
}
