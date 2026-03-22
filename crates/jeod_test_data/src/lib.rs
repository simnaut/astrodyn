pub mod euler_test;
pub mod gravity_verif;
pub mod orbital_data;
pub mod orbital_init;
pub mod reference_state;

/// Compile-time anchor: the directory containing this crate's Cargo.toml.
/// This is `<workspace>/crates/jeod_test_data`.
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// Get the JEOD root path.
///
/// Resolution order:
/// 1. `JEOD_PATH` environment variable (if set at runtime)
/// 2. `<workspace>/../jeod` (sibling of the repo checkout)
///
/// The workspace root is derived at compile time from this crate's
/// `CARGO_MANIFEST_DIR` (`<workspace>/crates/jeod_test_data`).
pub fn jeod_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("JEOD_PATH") {
        return std::path::PathBuf::from(p);
    }

    // MANIFEST_DIR = <workspace>/crates/jeod_test_data
    // workspace root = <workspace>  (two parents up)
    // JEOD sibling   = <workspace>/../jeod
    let workspace_root = std::path::Path::new(MANIFEST_DIR)
        .parent() // crates/
        .and_then(|p| p.parent()); // workspace root

    if let Some(root) = workspace_root {
        let sibling = root.join("../jeod");
        if sibling.exists() {
            return sibling;
        }
    }

    // Last resort — unlikely to be correct but fail visibly downstream.
    std::path::PathBuf::from("../jeod")
}
