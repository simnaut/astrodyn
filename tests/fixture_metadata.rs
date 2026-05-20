//! Fixture-metadata integrity test.
//!
//! Asserts that every committed extracted fixture under `crates/*/test_data/`
//! carries a JEOD-provenance sidecar (`<name>.json` or `<name>.meta.json`)
//! recording the upstream source path, the JEOD version, the JEOD commit
//! SHA, an extraction timestamp, and — for binary / canonical files — the
//! size and SHA-256 of the produced fixture. The sidecar fields are then
//! checked against the actual file bytes, so a future regen that drops or
//! desynchronises the metadata fails CI.
//!
//! The set of fixtures audited is hand-rolled (not directory-walked) so a
//! reviewer can read the test and see exactly which fixtures the supply-
//! chain story covers. Adding a new fixture to a regen binary requires
//! adding a matching entry here — same shape as the `extract_*` change
//! that produced it.
//!
//! Runs under `cargo nextest run --workspace -E 'not test(tier3_)'`.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// One audited fixture: the data file plus its sidecar.
struct AuditedFixture {
    /// Data file relative to workspace root.
    data: &'static str,
    /// Sidecar JSON relative to workspace root.
    sidecar: &'static str,
    /// Field name in the sidecar that records the data file's byte count.
    ///
    /// `None` means the sidecar carries no size field (the sidecar exists
    /// only for provenance, e.g. for a JSON fixture whose contents and
    /// metadata are merged into one document).
    size_field: Option<&'static str>,
    /// Field name in the sidecar that records the data file's SHA-256.
    ///
    /// `None` means the sidecar carries no hash field (same rationale as
    /// `size_field`).
    sha256_field: Option<&'static str>,
}

/// All audited fixtures. Order is alphabetical by sidecar path to keep
/// review diffs stable.
const FIXTURES: &[AuditedFixture] = &[
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/gemt1.bin",
        sidecar: "crates/astrodyn_gravity/test_data/gravity/gemt1.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/ggm02c.bin",
        sidecar: "crates/astrodyn_gravity/test_data/gravity/ggm02c.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/ggm05c.bin",
        sidecar: "crates/astrodyn_gravity/test_data/gravity/ggm05c.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/grav_geospherical_verif_out.txt",
        sidecar:
            "crates/astrodyn_gravity/test_data/gravity/grav_geospherical_verif_out.txt.meta.json",
        size_field: Some("reference_file_bytes"),
        sha256_field: Some("reference_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/mars_mro110b2.bin",
        sidecar: "crates/astrodyn_gravity/test_data/gravity/mars_mro110b2.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/moon_grail150.bin",
        sidecar: "crates/astrodyn_gravity/test_data/gravity/moon_grail150.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/moon_lp150q.bin",
        sidecar: "crates/astrodyn_gravity/test_data/gravity/moon_lp150q.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_gravity/test_data/gravity/sun_spherical.bin",
        sidecar: "crates/astrodyn_gravity/test_data/gravity/sun_spherical.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        // Inline metadata (the JSON is both data and its own sidecar);
        // no size/hash field to check, just presence of audit fields.
        data: "crates/astrodyn_planet/test_data/planet_pfixposn_seeds.json",
        sidecar: "crates/astrodyn_planet/test_data/planet_pfixposn_seeds.json",
        size_field: None,
        sha256_field: None,
    },
    AuditedFixture {
        data: "crates/astrodyn_time/test_data/Leap_Second.dat",
        sidecar: "crates/astrodyn_time/test_data/Leap_Second.dat.meta.json",
        size_field: Some("reference_file_bytes"),
        sha256_field: Some("reference_file_sha256"),
    },
    AuditedFixture {
        // Body-init bundles: inline metadata, no separate sidecar.
        data: "crates/astrodyn_verif_jeod/test_data/body_init/iss.json",
        sidecar: "crates/astrodyn_verif_jeod/test_data/body_init/iss.json",
        size_field: None,
        sha256_field: None,
    },
    AuditedFixture {
        data: "crates/astrodyn_verif_jeod/test_data/body_init/iss_mass.py",
        sidecar: "crates/astrodyn_verif_jeod/test_data/body_init/iss_mass.py.meta.json",
        size_field: Some("reference_file_bytes"),
        sha256_field: Some("reference_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_verif_jeod/test_data/body_init/sts_114.json",
        sidecar: "crates/astrodyn_verif_jeod/test_data/body_init/sts_114.json",
        size_field: None,
        sha256_field: None,
    },
    AuditedFixture {
        // Inline metadata: the JSON is the data file.
        data: "crates/astrodyn_verif_jeod/test_data/jeod_validation/euler_cases.json",
        sidecar: "crates/astrodyn_verif_jeod/test_data/jeod_validation/euler_cases.json",
        size_field: None,
        sha256_field: None,
    },
    AuditedFixture {
        data: "crates/astrodyn_verif_jeod/test_data/jeod_validation/orbital_vectors.bin",
        sidecar: "crates/astrodyn_verif_jeod/test_data/jeod_validation/orbital_vectors.json",
        size_field: Some("binary_file_bytes"),
        sha256_field: Some("binary_file_sha256"),
    },
    AuditedFixture {
        data: "crates/astrodyn_verif_nesc/test_data/cc8_nrho_reference.csv",
        sidecar: "crates/astrodyn_verif_nesc/test_data/cc8_nrho_reference.json",
        size_field: Some("reference_file_bytes"),
        sha256_field: Some("reference_file_sha256"),
    },
];

/// Required audit-trail fields on every sidecar. Sidecars referencing a
/// JEOD source must record the version + commit SHA captured at regen
/// time. The NESC sidecar swaps `jeod_*` for `nesc_release`; this list
/// is the lowest common denominator (`generated_utc` is universal).
const REQUIRED_AUDIT_FIELDS: &[&str] = &["generated_utc"];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_string(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}. (fixture-metadata audit)", path.display()))
}

fn read_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}. (fixture-metadata audit)", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Extract a `"key": <value>` from a flat JSON string. Returns the raw
/// value text (with surrounding whitespace trimmed) including quotes for
/// string values; the caller strips quotes as appropriate. Returns
/// `None` when the key is absent.
///
/// This is a deliberately tiny parser — sufficient for the flat
/// audit-metadata layout, intentionally not a full JSON implementation.
/// Every fixture sidecar in this audit is hand-written or emitted by an
/// extractor we control, so a heavyweight `serde_json` dep is not
/// warranted.
fn json_field<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let start = s.find(&needle)?;
    let after_key = start + needle.len();
    let after_colon_rel = s[after_key..].find(':')?;
    let val_start = after_key + after_colon_rel + 1;
    let val_slice = s[val_start..].trim_start();

    let mut depth_obj = 0;
    let mut depth_arr = 0;
    let mut in_str = false;
    let mut escape = false;
    let bytes = val_slice.as_bytes();
    let mut end = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        match b {
            b'\\' if in_str => escape = true,
            b'"' => in_str = !in_str,
            b'{' if !in_str => depth_obj += 1,
            b'}' if !in_str => {
                if depth_obj == 0 {
                    end = i;
                    break;
                }
                depth_obj -= 1;
            }
            b'[' if !in_str => depth_arr += 1,
            b']' if !in_str => {
                if depth_arr == 0 {
                    end = i;
                    break;
                }
                depth_arr -= 1;
            }
            b',' if !in_str && depth_obj == 0 && depth_arr == 0 => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    Some(val_slice[..end].trim())
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

#[test]
fn fixture_metadata_is_present_and_consistent() {
    let root = workspace_root();
    let mut failures: Vec<String> = Vec::new();

    for fx in FIXTURES {
        let data_path = root.join(fx.data);
        let sidecar_path = root.join(fx.sidecar);

        if !data_path.exists() {
            failures.push(format!(
                "missing fixture: {}\n  Sidecar references this file but it is not committed.",
                data_path.display()
            ));
            continue;
        }
        if !sidecar_path.exists() {
            failures.push(format!(
                "missing sidecar: {}\n  Every committed extracted fixture must carry a \
                 provenance sidecar (jeod_version + jeod_commit + generated_utc + \
                 size + sha256). Update the extractor and commit the regenerated sidecar.",
                sidecar_path.display()
            ));
            continue;
        }

        let sidecar_text = read_string(&sidecar_path);

        for field in REQUIRED_AUDIT_FIELDS {
            if json_field(&sidecar_text, field).is_none() {
                failures.push(format!(
                    "{}: missing required audit field `{field}`. \
                     The extractor must emit jeod_version + jeod_commit + generated_utc.",
                    sidecar_path.display()
                ));
            }
        }

        // Either {jeod_version + jeod_commit} or {nesc_release} must be
        // present; the latter is the NESC-track equivalent.
        let has_jeod = json_field(&sidecar_text, "jeod_version").is_some()
            && json_field(&sidecar_text, "jeod_commit").is_some();
        let has_nesc = json_field(&sidecar_text, "nesc_release").is_some();
        if !has_jeod && !has_nesc {
            failures.push(format!(
                "{}: missing upstream-release identifier. \
                 Expected either (jeod_version + jeod_commit) or nesc_release.",
                sidecar_path.display()
            ));
        }

        if let (Some(size_field), Some(sha_field)) = (fx.size_field, fx.sha256_field) {
            let bytes = read_bytes(&data_path);
            let observed_size = bytes.len() as u64;
            let observed_sha = sha256_hex(&bytes);

            let recorded_size: u64 = json_field(&sidecar_text, size_field)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: missing `{size_field}` field. \
                         The extractor must record the produced file's byte count.",
                        sidecar_path.display()
                    )
                })
                .parse()
                .unwrap_or_else(|_| {
                    panic!(
                        "{}: `{size_field}` is not an integer",
                        sidecar_path.display()
                    )
                });
            let recorded_sha =
                strip_quotes(json_field(&sidecar_text, sha_field).unwrap_or_else(|| {
                    panic!(
                        "{}: missing `{sha_field}` field. \
                         The extractor must record the produced file's SHA-256.",
                        sidecar_path.display()
                    )
                }))
                .to_string();

            if recorded_size != observed_size {
                failures.push(format!(
                    "{}: size mismatch for {}: sidecar says {recorded_size}, file is {observed_size} bytes. \
                     Either the binary was regenerated without updating the sidecar, or vice versa.",
                    sidecar_path.display(),
                    data_path.display(),
                ));
            }
            if recorded_sha != observed_sha {
                failures.push(format!(
                    "{}: SHA-256 mismatch for {}: sidecar says {recorded_sha}, observed {observed_sha}. \
                     The binary fixture and its recorded provenance have drifted apart. \
                     Re-run the extract_* regen binary or update the sidecar to match.",
                    sidecar_path.display(),
                    data_path.display(),
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "fixture-metadata audit found {} issue(s):\n  - {}",
        failures.len(),
        failures.join("\n  - "),
    );
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn json_field_reads_string_value() {
        let s = r#"{"a": "hello", "b": 42}"#;
        assert_eq!(json_field(s, "a"), Some("\"hello\""));
    }

    #[test]
    fn json_field_reads_integer_value() {
        let s = r#"{"a": "hello", "b": 42}"#;
        assert_eq!(json_field(s, "b"), Some("42"));
    }

    #[test]
    fn json_field_handles_trailing_field() {
        let s = "{\n  \"k\": \"v\"\n}\n";
        assert_eq!(strip_quotes(json_field(s, "k").unwrap()), "v");
    }

    #[test]
    fn json_field_missing_returns_none() {
        assert!(json_field("{}", "missing").is_none());
    }
}
