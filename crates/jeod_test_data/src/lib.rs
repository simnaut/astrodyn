pub mod euler_test;
pub mod gravity_verif;
pub mod orbital_data;
pub mod orbital_init;
pub mod reference_state;

/// Get the JEOD root path from env or default.
pub fn jeod_path() -> std::path::PathBuf {
    std::env::var("JEOD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            // Try relative to manifest dir, then ../jeod
            let manifest =
                std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
            let p = std::path::Path::new(&manifest).join("../../jeod");
            if p.exists() {
                p
            } else {
                std::path::PathBuf::from("../jeod")
            }
        })
}
