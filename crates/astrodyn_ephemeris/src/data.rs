//! Catalog of ephemeris and orientation kernels distributed alongside the crate.
//!
//! Each [`KernelSpec`] names a binary asset hosted on the project's
//! `kernels-v1` GitHub Release. Call [`load`] to obtain the raw bytes,
//! which can be passed to [`Ephemeris::from_bsp_bytes`] /
//! [`Ephemeris::load_bpc_bytes`] (or via the higher-level
//! `astrodyn::recipes::ephemeris` recipes).
//!
//! ## Lookup order
//!
//! 1. `$ASTRODYN_EPHEMERIS_KERNELS_DIR/<name>` — explicit override for
//!    air-gapped CI or vendored builds.
//! 2. `$CARGO_MANIFEST_DIR/assets/<name>` — the committed in-tree
//!    kernels. Resolves only for workspace builds; the published
//!    `.crate` does not ship `assets/`.
//! 3. `<cache>/astrodyn-ephemeris/<name>` — `$XDG_CACHE_HOME` or
//!    `$HOME/.cache` (with a `tmpdir` fallback when neither is set).
//! 4. (`fetch` feature, default-on) HTTPS GET `spec.url`, SHA-256
//!    verify, atomic-write to the cache, return bytes.
//!
//! ## Offline operation
//!
//! Disable the default `fetch` feature and pre-populate either step (1)
//! or (3); [`load`] will never touch the network.
//!
//! [`Ephemeris::from_bsp_bytes`]: crate::Ephemeris::from_bsp_bytes
//! [`Ephemeris::load_bpc_bytes`]: crate::Ephemeris::load_bpc_bytes

use std::path::{Path, PathBuf};

use crate::EphemerisError;

/// Metadata for a single distributed kernel asset.
#[derive(Clone, Copy)]
pub struct KernelSpec {
    /// File name on disk (also the asset name in the GitHub Release).
    pub name: &'static str,
    /// Direct-download URL for the asset, pinned to the `kernels-v1` tag
    /// so kernel data lifecycle is decoupled from the source-crate
    /// `v*` release lifecycle.
    pub url: &'static str,
    /// Lowercase hex SHA-256 of the kernel bytes. Verified after fetch.
    pub sha256: &'static str,
    /// Expected file size in bytes. Verified after every read.
    pub bytes: u64,
}

/// JPL DE421 planetary ephemeris (1900–2050, J2000 ICRF, ~17 MB).
///
/// Covers Sun, Moon, planets, and Earth–Moon barycenter. Used by the
/// JEOD SIM_dyncomp Tier 3 baselines.
pub const DE421: KernelSpec = KernelSpec {
    name: "de421.bsp",
    url: "https://github.com/simnaut/astrodyn/releases/download/kernels-v1/de421.bsp",
    sha256: "08b20db2ae22488650641c5a9033e5bfda4b1c4b440cfeaf20f621cfa18ecdb3",
    bytes: 16_790_528,
};

/// JPL DE440 short-subset planetary ephemeris (1849–2150, J2000 ICRF, ~31 MB).
///
/// The `de440s.bsp` shipped by NAIF
/// (<https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/>) —
/// the truncated subset is sufficient for every modern simulation epoch
/// the project currently targets and is two orders of magnitude smaller
/// than the full `de440.bsp`. Pinned to the generation that the NASA
/// NESC GN&C Lunar Check Cases (NESC-RP-23-01853) specify.
pub const DE440: KernelSpec = KernelSpec {
    name: "de440.bsp",
    url: "https://github.com/simnaut/astrodyn/releases/download/kernels-v1/de440.bsp",
    sha256: "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2",
    bytes: 32_726_016,
};

/// Moon principal-axes orientation kernel (1900–2050, ~1.7 MB).
///
/// Required by consumers that need the Moon's physical orientation
/// (libration). The SPK alone gives Moon position/velocity but not
/// body-fixed attitude.
pub const MOON_PA: KernelSpec = KernelSpec {
    name: "moon_pa_de421_1900-2050.bpc",
    url:
        "https://github.com/simnaut/astrodyn/releases/download/kernels-v1/moon_pa_de421_1900-2050.bpc",
    sha256: "656f90616403d75a75f0cd6c8830fc5b44f8cb4facb5ccb8915e752b397520cf",
    bytes: 1_770_496,
};

/// Locate and return the bytes for `spec`. See the module-level docs
/// for the four-step lookup order.
///
/// The size of every returned blob is checked against `spec.bytes`.
/// Full SHA-256 verification is run only after a fresh network fetch —
/// local files (workspace `assets/` or user cache) are trusted by size
/// alone to keep `load` cheap on hot paths.
pub fn load(spec: &KernelSpec) -> Result<Vec<u8>, EphemerisError> {
    if let Some(dir) = std::env::var_os("ASTRODYN_EPHEMERIS_KERNELS_DIR") {
        let path = PathBuf::from(dir).join(spec.name);
        if path.is_file() {
            return read_local(&path, spec);
        }
    }

    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(spec.name);
    if manifest_path.is_file() {
        return read_local(&manifest_path, spec);
    }

    let cache_path = cache_dir().join(spec.name);
    if cache_path.is_file() {
        return read_local(&cache_path, spec);
    }

    #[cfg(feature = "fetch")]
    {
        let bytes = fetch::download(spec)?;
        let _ = atomic_write(&cache_path, &bytes);
        Ok(bytes)
    }

    #[cfg(not(feature = "fetch"))]
    Err(EphemerisError::LoadError(format!(
        "kernel `{}` not found in $ASTRODYN_EPHEMERIS_KERNELS_DIR, \
         $CARGO_MANIFEST_DIR/assets, or {}, and the `fetch` feature is disabled. \
         Pre-populate one of these locations or rebuild with `--features fetch`.",
        spec.name,
        cache_path.display(),
    )))
}

fn read_local(path: &Path, spec: &KernelSpec) -> Result<Vec<u8>, EphemerisError> {
    let bytes = std::fs::read(path)
        .map_err(|e| EphemerisError::LoadError(format!("reading {}: {e}", path.display())))?;
    if bytes.len() as u64 != spec.bytes {
        return Err(EphemerisError::LoadError(format!(
            "{}: expected {} bytes, got {} — file is truncated or replaced; \
             clear it and re-run with the `fetch` feature, or restore from `kernels-v1`",
            path.display(),
            spec.bytes,
            bytes.len(),
        )));
    }
    Ok(bytes)
}

fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(d).join("astrodyn-ephemeris");
    }
    if let Some(d) = std::env::var_os("HOME") {
        return PathBuf::from(d).join(".cache").join("astrodyn-ephemeris");
    }
    std::env::temp_dir().join("astrodyn-ephemeris")
}

#[cfg(feature = "fetch")]
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("partial");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(feature = "fetch")]
mod fetch {
    use std::io::Read;

    use sha2::{Digest, Sha256};

    use super::{EphemerisError, KernelSpec};

    pub fn download(spec: &KernelSpec) -> Result<Vec<u8>, EphemerisError> {
        let resp = ureq::get(spec.url)
            .call()
            .map_err(|e| EphemerisError::LoadError(format!("GET {}: {e}", spec.url)))?;
        // `with_capacity` is a hint only; the explicit length check below is
        // the authoritative validation, so a truncating cast on 32-bit
        // targets degrades to a re-allocation rather than a wrong result.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "capacity hint; length is validated below"
        )]
        let mut bytes = Vec::with_capacity(spec.bytes as usize);
        resp.into_reader().read_to_end(&mut bytes).map_err(|e| {
            EphemerisError::LoadError(format!("reading response body from {}: {e}", spec.url))
        })?;
        if bytes.len() as u64 != spec.bytes {
            return Err(EphemerisError::LoadError(format!(
                "{}: expected {} bytes from {}, got {}",
                spec.name,
                spec.bytes,
                spec.url,
                bytes.len(),
            )));
        }
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if got != spec.sha256 {
            return Err(EphemerisError::LoadError(format!(
                "{}: SHA-256 mismatch (expected {}, got {}) — refusing to cache \
                 a kernel that doesn't match the `kernels-v1` release manifest",
                spec.name, spec.sha256, got,
            )));
        }
        Ok(bytes)
    }
}
