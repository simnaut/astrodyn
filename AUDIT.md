# bevy_jeod Codebase Audit

**Date:** 2026-03-26
**Scope:** All 14 crates (87 Rust source files)
**Focus:** Pitfalls, silent misconfigurations, inconsistencies, poor practices

---

## Executive Summary

The codebase is well-structured and faithfully ports JEOD conventions. The two-layer
architecture (physics in `jeod_*`, Bevy glue in `bevy_jeod_*`) is respected throughout.
All numerics use `f64`/`DVec3`/`DQuat` — no `f32` or Bevy `Transform` contamination.
Quaternion conventions are correctly handled at boundaries.

However, the audit uncovered **6 critical**, **8 high**, and **12 medium** severity
issues, mostly in three categories:

1. **`debug_assert!` used for runtime safety checks** — silently becomes a no-op in
   release builds, allowing NaN/Inf propagation or panics
2. **Silent degradation in Bevy systems** — missing components cause physics to be
   skipped or computed incorrectly without errors
3. **Test tolerances 2–5× looser than observed values** — hiding potential regressions

---

## Critical Issues

### C1. `debug_assert!` guards critical invariants (11 call sites)

**Severity:** CRITICAL
**Pattern:** `debug_assert!` is used to guard against division-by-zero, zero-length
vectors, and singular matrices. In release builds, these checks are compiled out.

| File | Line | Guard |
|------|------|-------|
| `jeod_math/src/quaternion.rs` | 86 | Zero quaternion normalization → NaN |
| `jeod_math/src/solar_beta.rs` | 25, 29 | Zero vector `.normalize()` → panic |
| `jeod_dynamics/src/rotational.rs` | 113 | Singular inertia → NaN angular accel |
| `jeod_dynamics/src/mass.rs` | 37 | Singular inertia tensor → Inf inverse |
| `jeod_dynamics/src/forces.rs` | 65 | Zero mass → Inf acceleration |
| `jeod_gravity/src/compute.rs` | 11, 95, 172 | Zero position → Inf gravity |
| `jeod_gravity/src/spherical_harmonics_calc_nonspherical.rs` | 128, 136 | Scratch degree mismatch; zero position |

**Impact:** In release mode, a body at the origin (e.g., before initialization) silently
produces `Inf`/`NaN` acceleration that propagates through integration, corrupting the
entire simulation state. No error, no panic — just wrong numbers.

**Recommendation:** Replace with runtime checks. For hot-path code (gravity inner loop),
use a one-time validation at system entry rather than per-call checks.

---

### C2. Rotational dynamics silently degrades to 3-DOF

**File:** `bevy_jeod_dynamics/src/systems.rs:110-128`

When `DynamicsConfig::rotational_dynamics = true` but the entity lacks `RotationalStateC`
or `MassPropertiesC`, the system silently falls through to the 3-DOF translational
integrator. No warning, no error.

```rust
if config.rotational_dynamics {
    if let (Some(ref mut rot), Some(mass_props)) = (&mut rot_state, &mass) {
        // ... 6-DOF path
        continue;
    }
    // SILENT FALLTHROUGH to 3-DOF
}
```

**Impact:** A user enables 6-DOF but forgets a component. The simulation runs without
rotational dynamics and produces plausible-looking (but wrong) results.

**Recommendation:** Add `warn_once!` when rotational_dynamics is true but required
components are missing.

---

### C3. Missing gravity source silently leaves stale acceleration

**File:** `bevy_jeod_gravity/src/systems.rs:32-39` and
`bevy_jeod_dynamics/src/systems.rs:98`

When a `GravityControl` references a source entity that doesn't exist or lacks
`GravitySourceC`, the gravity computation silently skips that source. The
`GravityAccelerationC` component retains its previous value (or default zeros).

**Impact:** A misconfigured entity reference causes gravity to silently vanish.
The body continues in a straight line with no indication of the problem.

---

### C4. TDB-TAI convergence fails near tai ≈ 0

**File:** `jeod_time/src/time_converter_tai_tdb.rs:43`

```rust
if tai.abs() > 0.0 && (dtai / tai).abs() < 1.0e-15 {
    break;
}
```

When `tai ≈ 0` (near simulation start), the relative convergence test is skipped.
The loop runs all 5 iterations and may return a suboptimal result.

**Impact:** Reduced ephemeris accuracy near simulation epoch.

**Recommendation:** Use absolute tolerance as fallback: `dtai.abs() < 1e-15`.

---

### C5. Empty leap second table panics with integer underflow

**File:** `jeod_time/src/leap_second.rs:85, 101`

```rust
let last = self.entries.len() - 1;  // panics if entries is empty
```

The public methods `tai_utc_at_tai_tjt` and `tai_utc_at_utc_tjt` have guards for empty
tables, but the private `find_index_for_tai` / `find_index_for_utc` do not. The
constructor `from_entries()` doesn't reject empty input.

**Recommendation:** Validate non-empty in constructor, or add guard in private methods.

---

### C6. `assert!` in leap second constructor compiled out in release

**File:** `jeod_time/src/leap_second.rs:19`

```rust
assert!(
    entries.windows(2).all(|w| w[0].0 <= w[1].0),
    "Leap second entries must be sorted by TJT"
);
```

This uses `assert!` (not `debug_assert!`), so it IS active in release — **this is fine.**
However, checking this: the `SphericalHarmonicsData::new()` at
`jeod_gravity/src/spherical_harmonics_gravity_source.rs:53-54` uses `assert!` correctly
too. The `MassProperties::new()` also uses `assert!`. Good.

**Correction:** On closer inspection, only the `debug_assert!` sites from C1 are
problematic. The `assert!` sites are correctly active in release. Downgrading C6 —
the constructor validation is fine.

---

## High-Severity Issues

### H1. `force_collection_system` silently skips bodies without `MassPropertiesC`

**File:** `bevy_jeod_dynamics/src/systems.rs:19-26`

The query `(&GravityAccelerationC, &MassPropertiesC, &mut TotalForceC)` means entities
without all three components are silently excluded. A body with gravity acceleration
but no mass properties has its `TotalForceC` never updated.

---

### H2. Non-spherical gravity with identity rotation continues silently

**File:** `bevy_jeod_dynamics/src/systems.rs:73-90`

When non-spherical gravity (degree > 0) is requested but `PlanetFixedRotationC` is
missing, the system logs a `warn_once!` but then uses `DMat3::IDENTITY` as the
planet-fixed rotation. This produces physically incorrect results — the gravity
field orientation is wrong. A single warning message (per process lifetime) is
insufficient for an error that silently corrupts every subsequent timestep.

**Recommendation:** Either skip the non-spherical contribution or return an error
component that downstream systems can check.

---

### H3. Leap second boundary ambiguity

**File:** `jeod_time/src/leap_second.rs:100-115`

The `find_index_for_utc()` function's boundary semantics are ambiguous. When
`utc_tjt` equals a leap second boundary exactly, it's unclear whether the old or new
offset applies. The early-exit guard (line 102-103 returns 0) and the boundary case
(line 105-106 returns `last`) use different conventions.

---

### H4. UT1-TAI offset assumes UT1-UTC ≈ 0

**File:** `jeod_time/src/simulation_time.rs:43-46`

The constructor sets `ut1_tai_offset = -tai_utc_s`, implicitly assuming UT1-UTC = 0.
For historical or far-future epochs, UT1-UTC can be ±0.9s, affecting GMST calculations.
The assumption is undocumented in the API.

---

### H5. Frame tree `add_child()` panics on invalid parent ID

**File:** `jeod_frames/src/frame_tree.rs:62-75`

No validation that `parent_id` is a valid index. If `parent_id >= nodes.len()`, the
function panics at `self.children[parent_id].push(id)`.

---

### H6. `RefFrameKind` stored but never enforced

**File:** `jeod_frames/src/frame_tree.rs:20-21`

The `RefFrameKind` enum (Inertial, PlanetFixed, Body) is stored in every frame node
but never matched on or validated. A user can create nonsensical frame hierarchies
(e.g., a Body frame as parent of a PlanetFixed frame) without warning.

---

### H7. No validation of rotation matrix / quaternion consistency

**File:** `jeod_frames/src/ref_frame_state.rs`

A `RotationalState` stores both `q_parent_this` and `t_parent_this` independently.
Nothing ensures they represent the same rotation. If they diverge due to a bug, the
error propagates silently through all frame transformations.

---

### H8. `TotalForce` mixes reference frames

**File:** `jeod_dynamics/src/forces.rs:35-39`

```rust
pub struct TotalForce {
    pub force: DVec3,  // N, in integration frame
    pub torque: DVec3, // N*m, in body frame
}
```

Force is in integration frame, torque is in body frame. This mixed-frame struct is
a bug waiting to happen when Phase 4 adds non-gravity torques. Any code that accumulates
both must remember to transform between frames, with no type-level enforcement.

---

### H9. Geodetic iteration can diverge

**File:** `jeod_math/src/geodetic.rs:129`

The Borkowski algorithm iteration divides by `d = 2*[cos(y0-w) - c*cos(2*y0)]`.
If `d ≈ 0`, the iteration diverges. The 20-iteration limit prevents infinite loops
but doesn't guarantee convergence. No error is reported on non-convergence.

---

### H10. Silent degree/order clamping in spherical harmonics

**File:** `jeod_gravity/src/spherical_harmonics_calc_nonspherical.rs:133-134`

```rust
let degree = degree.min(data.degree);
let order = order.min(data.order).min(degree);
```

When a user requests degree=360 but the loaded model only has degree=40, the code
silently clamps to 40 with no warning. The same applies to gradient degree/order
(lines 147-156). A user may believe they're running a high-fidelity gravity model
when they're actually running at much lower fidelity.

**Recommendation:** Log a warning when clamping occurs, or return an error.

---

### H11. Coefficient loading panics on I/O errors

**File:** `jeod_gravity/src/coefficients.rs:21, 57-60, 118-119, 125, 136, 141, 146`

Multiple `unwrap()` and `panic!()` calls in the coefficient loading path. File not
found, parse errors, and I/O failures all produce unrecoverable panics rather than
`Result` errors. While this is startup-only code, it makes the library hostile to
embed in applications that need graceful error handling.

---

## Medium-Severity Issues

### M1. Test tolerances 2–5× looser than observed values

| Test | Tolerance | Observed | Margin |
|------|-----------|----------|--------|
| Tier 3 SH position (`tier3_spherical_harmonics.rs:181`) | 0.5 m | 0.12–0.20 m | 2.5–4× |
| J2 regression rate (`j2_regression.rs:100`) | 1% | ~0.1% | 10× |
| 6-DOF quaternion error (`tier3_sixdof.rs:246`) | 0.01 rad | ~0.001 rad | 10× |
| Energy conservation per-step (`tier3_trajectory.rs:78`) | 1e-7 | <1e-8 | 10× |

**Impact:** Regressions could increase errors by 2–5× before any test catches them.

---

### M2. Gimbal lock skip count not asserted

**File:** `jeod_math/tests/tier3_euler_angles.rs:174-185`

The test silently skips timesteps near gimbal lock via `continue`. The skip count
is logged but not asserted. If attitude propagation drifts, more points could be
skipped, masking failures.

---

### M3. Julian date precision loss in ephemeris

**File:** `jeod_ephemeris/src/ephemeris.rs:41`

```rust
let tdb_s_since_j2000 = (tdb_jd - 2_451_545.0) * 86_400.0;
```

For epochs far from J2000, the subtraction loses precision. Both operands are ~10^6,
so the difference loses ~3 digits of precision.

---

### M4. `SimulationTime::advance()` accepts NaN/Inf/negative dt

**File:** `jeod_time/src/simulation_time.rs:95-100`

No validation on the `dt` parameter. Passing NaN silently corrupts all time scales.

---

### M5. TDB-TT accuracy claim untested

**File:** `jeod_time/src/time_converter_tai_tdb.rs:5-18`

The docstring claims "accurate to ±0.1 microseconds" but no test validates this.

---

### M6. `PlanetShape` allows invalid construction

**File:** `jeod_planet/src/planet.rs:6-18`

No validation that `flat_coeff ∈ [0, 1)`, `r_eq >= r_pol`, or that `r_pol` is
consistent with `r_eq * (1 - flat_coeff)`. The `r_pol` and `flat_coeff` are
independently specified (two sources of truth).

---

### M7. Polar motion explicitly disabled

**File:** `jeod_frames/src/rotation_j2000.rs:50-51`

The Earth rotation model omits polar motion. This is documented in a code comment
but not surfaced to API users. Affects accuracy at the ~1e-3 m level for LEO.

---

### M8. Hardcoded time constant in rotation_j2000

**File:** `jeod_frames/src/rotation_j2000.rs:83`

```rust
let tt_centuries = (tt_tjt - 11544.5) / 36525.0;
```

The J2000 epoch in TJT (11544.5) is hardcoded, not imported from `jeod_time`. The
`jeod_time` crate uses different epoch constants, creating a consistency risk.

---

### M9. Missing TJT validation in leap second lookups

**File:** `jeod_time/src/leap_second.rs:38, 51`

The public methods accept any `f64`. NaN or Inf values produce incorrect indices
without error.

---

### M10. Ephemeris errors are opaque

**File:** `jeod_ephemeris/src/ephemeris.rs:46-49`

ANISE errors are stringified, losing semantic information. Callers can't distinguish
"epoch out of range" from "frame not available" without parsing error strings.

---

### M11. Frame tree assumes connected graph

**File:** `jeod_frames/src/frame_tree.rs:118-137`

`find_common_ancestor()` panics if frames don't share a root. Disconnected frame
trees (possible via API misuse) cause unrecoverable panics.

---

### M12. Computational independence violations in tests

**File:** `jeod_math/tests/tier3_orbital_elements.rs:157` and
`jeod_dynamics/tests/tier3_frame_propagation.rs:202-268`

Some Tier 3 tests use JEOD CSV output as both input and validation. Per CLAUDE.md,
JEOD reference data should be used **only** for comparison, never as computation input.

---

## Positive Findings

- **Two-layer architecture**: Fully compliant. No physics in `bevy_jeod_*` crates.
- **Precision**: All `f64`/`DVec3`/`DMat3`/`DQuat`. Zero `f32` contamination.
- **Quaternion conventions**: JEOD scalar-first ↔ glam boundary correctly handled.
- **System ordering**: `JeodSet` enum correctly chains the integration loop.
- **Spherical harmonics**: Faithful port of Gottlieb algorithm with underflow guards.
- **RK4 integration**: Both 3-DOF and 6-DOF paths are correctly implemented with
  multi-stage gravity re-evaluation.
- **Kepler solvers**: Safe for valid elliptic/hyperbolic inputs (denominators proven
  positive).
- **Euler angle extraction**: All 12 sequences with gimbal lock detection.
- **Test coverage**: Comprehensive unit tests, meaningful Tier 2/3 cross-validation.
- **Scratch buffer reuse**: Avoids per-call heap allocation in gravity inner loop.

---

## Recommendations (Priority Order)

### Immediate

1. **Replace `debug_assert!` with runtime checks** in the 11 call sites listed in C1.
   For the gravity hot path, validate once at system entry (not per RK4 stage).

2. **Add `warn_once!` for silent 6-DOF fallthrough** (C2). A one-line warning prevents
   hours of debugging.

3. **Tighten test tolerances** to 1.5× observed values (M1). Current tolerances allow
   2–10× regressions.

### Soon

4. **Validate frame tree operations** — bounds-check `parent_id` in `add_child()` (H5).

5. **Assert gimbal lock skip counts** in Euler angle tests (M2).

6. **Use absolute convergence fallback** in TDB-TAI conversion (C4).

7. **Document the UT1-UTC ≈ 0 assumption** in `SimulationTime` (H4).

### When Convenient

8. Enforce `RefFrameKind` semantics or remove the field (H6).

9. Add `is_consistent()` method for `RotationalState` (H7).

10. Import epoch constants from `jeod_time` instead of hardcoding (M8).

11. Add input validation to `SimulationTime::advance()` (M4).

12. Refactor computational-independence violations in Tier 3 tests (M12).
