pub mod euler_test;
pub mod gravity_verif;
pub mod orbital_data;
pub mod orbital_init;
pub mod reference_state;

/// Get the JEOD root path.
///
/// Resolution order:
/// 1. `JEOD_PATH` environment variable (if set)
/// 2. `<workspace_root>/../jeod` (sibling of the repo checkout)
///
/// The workspace root is derived from `CARGO_MANIFEST_DIR` (this crate lives
/// at `<workspace>/crates/jeod_test_data`).
pub fn jeod_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("JEOD_PATH") {
        return std::path::PathBuf::from(p);
    }

    // Derive workspace root from this crate's manifest directory.
    // CARGO_MANIFEST_DIR = <workspace>/crates/jeod_test_data
    // workspace root     = <workspace>
    // JEOD sibling       = <workspace>/../jeod
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let workspace_root = std::path::Path::new(&manifest)
        .parent() // crates/
        .and_then(|p| p.parent()); // workspace root

    if let Some(root) = workspace_root {
        let sibling = root.join("../jeod");
        if sibling.exists() {
            return sibling;
        }
    }

    // Last resort (unlikely to be correct, but matches legacy behavior).
    std::path::PathBuf::from("../jeod")
}
