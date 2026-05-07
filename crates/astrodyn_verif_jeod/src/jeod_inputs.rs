//! Verbatim JEOD source-file fixtures committed under
//! `crates/astrodyn_verif_jeod/test_data/jeod_inputs/<jeod-relative-path>`.
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
//! `crates/astrodyn_verif_jeod/test_data/jeod_inputs/verif/SIM_dyncomp/Modified_data/mass.py`.
//! The mirror keeps the audit trail trivial — `diff` against any JEOD
//! checkout to confirm we have the upstream contents.
//!
//! ## Regenerate after a JEOD upgrade
//!
//! See `crates/astrodyn_verif_jeod/test_data/jeod_inputs/README.md` for the `cp` recipe. New
//! fixtures are added by copying from `$JEOD_HOME` into the mirror
//! directory and committing the result.

use std::path::{Component, Path, PathBuf};

/// Resolve a JEOD-relative path against the committed
/// `crates/astrodyn_verif_jeod/test_data/jeod_inputs/` mirror.
///
/// `relative` must be a JEOD-source-relative path
/// (e.g. `"verif/SIM_dyncomp/S_define"` or
/// `"models/dynamics/derived_state/verif/SIM_NED/Modified_data/date_and_time.py"`).
/// Panics if `relative` is absolute or contains a `..` segment — the
/// resolved path must stay rooted under `crates/astrodyn_verif_jeod/test_data/jeod_inputs/`.
/// Also panics if the resolved path does not exist; committed fixtures
/// must always be present.
pub fn path(relative: &str) -> PathBuf {
    let rel = Path::new(relative);
    assert!(
        rel.is_relative(),
        "JEOD input path must be JEOD-source-relative, got absolute path: {relative}"
    );
    assert!(
        !rel.components()
            .any(|c| matches!(c, Component::ParentDir | Component::RootDir)),
        "JEOD input path must stay rooted under crates/astrodyn_verif_jeod/test_data/jeod_inputs/; \
         '..' / root segments are rejected: {relative}"
    );

    let p = workspace_root()
        .join("crates/astrodyn_verif_jeod/test_data/jeod_inputs")
        .join(rel);
    assert!(
        p.exists(),
        "JEOD input fixture not found at {}. \
         If you added a new sim, copy the file from $JEOD_HOME/{relative} \
         into crates/astrodyn_verif_jeod/test_data/jeod_inputs/{relative} and commit it. \
         See the module-level docs and crates/astrodyn_verif_jeod/test_data/jeod_inputs/README.md.",
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
    // Fallback: two parents up from `crates/astrodyn_verif_jeod/`.
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

    #[test]
    #[should_panic(expected = "must be JEOD-source-relative, got absolute path")]
    fn rejects_absolute_path() {
        let _ = path("/etc/passwd");
    }

    #[test]
    #[should_panic(expected = "'..' / root segments are rejected")]
    fn rejects_parent_dir_traversal() {
        let _ = path("verif/SIM_dyncomp/../../../etc/passwd");
    }
}
