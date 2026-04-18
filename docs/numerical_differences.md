# Irreducible Numerical Differences vs JEOD

This catalog documents **legitimate** floating-point differences between this
Rust port and JEOD's C++ reference. These are not bugs — they result from
building two independent implementations that share the same physics but
evaluate it on different toolchains, libraries, and interpolation paths. The
doc exists so a future engineer who sees, for example, a ~10 arcsecond Sun
offset in a Tier 3 test recognises expected convergence noise rather than a
regression. When a discrepancy can be traced to a formula, constant, or sign
convention it is a bug and must be fixed (see `CLAUDE.md` "No Half-Baked
Implementations"); only differences from the sources below should widen a
tolerance. Root-cause analysis for the 8-hour trajectory residuals is captured
in simnaut/bevy_jeod#6, the DE421 path in #27, and the SRP thermal residual in
#13 / #114.

## GCC vs LLVM floating-point (trajectory and RNP)

- **Symptom / magnitude:** 0.12–0.32 m position and ~4e-4 m/s velocity over an
  8-hour ISS orbit at dt=0.03125 s (921 600 RK4 steps); ~1.96e-11 element-wise
  on the composed `T_parent_this` RNP matrix; ~4.4e-8 rad attitude quaternion
  and ~3e-18 rad/s angular velocity.
- **Where observed:** simnaut/bevy_jeod#6; tolerances in
  `crates/jeod_runner/tests/tier3_sim_dyncomp_run3.rs:189` (RUN_3A: 4×4 SH +
  RNP) and `:201` (RUN_3B: 8×8 SH + RNP).
- **Root cause:** Identical formulae, different toolchain. GCC may emit
  single-rounded `fma(a,b,c)` where LLVM emits a double-rounded `a*b + c`;
  compilers reorder commutative operations at the same `-O` level; and glibc
  vs LLVM libm differ by up to 1 ULP on `sin`/`cos`/`atan2`. Per-operation
  these are ~2.2e-16 (1 ULP at f64) — the GAST polynomial alone dominates the
  RNP residual at ~100 ULPs, and coherent drift through 10^6 RK4 stages
  accumulates to decimetre-scale trajectory error.
- **Why irreducible:** Both implementations evaluate the same mathematical
  expression. Closing the gap requires freezing a single binary toolchain
  across JEOD and our crate — which this project explicitly rejects in favour
  of computational independence. Error scales as ~dt⁴ for large dt and
  plateaus at the FP noise floor for small dt, confirming the mechanism.

## DE421 ephemeris interpolation (Anise vs JEOD DE4xx reader)

- **Symptom / magnitude:** ~10 arcseconds directional offset on the Sun
  vector, with bounded solar-beta residuals at the
  `tier3_sim_solar_beta_edge.rs` tolerances (1.892e-5 / 3.446e-5 rad);
  the ~5e-5 to 1.5e-4 rad/day growth originally reported in #27 was
  Phase 4b historical observation and no longer holds at current
  tolerances. Also: constant ~0.31 m offset on Moon-centered state
  after Apollo frame switch; ~0.97 m position over a 7-day Earth-Moon
  Clementine trajectory; ~2 m position drift on the SIM_Tides 23-day
  case.
- **Where observed:** simnaut/bevy_jeod#27; commentary in
  `crates/jeod_runner/tests/tier3_sim_torque_simple.rs:20`,
  `tier3_apollo8_frame_switch.rs:334`, `tier3_sim_earth_moon.rs:261`,
  `tier3_sim_tide_verif.rs:281`, `tier3_sim_solar_beta_edge.rs`
  (beta_tol 1.892e-5 / 3.446e-5 rad).
- **Root cause:** JEOD uses `cspice`-derived DE4xx Chebyshev evaluators
  linked into its C++ binary, and its default kernel is DE405. We use
  [Anise](https://github.com/nyx-space/anise), a pure-Rust SPICE reader
  that loads either DE405 or DE421 BSP kernels and evaluates the same
  Chebyshev polynomials. When both sides use the same kernel (e.g.
  Apollo 8 with `de405.bsp`), the coefficient streams are identical on
  the overlapping span and the residual reduces to Anise vs cspice
  differences in recurrence formulation and intermediate rounding.
  When one side uses DE405 and the other DE421 (e.g. Earth-Moon and
  SIM_Tides, where we have no DE405 LE BSP), there is an additional
  ephemeris-model mismatch on top of the interpolation difference.
- **Why irreducible:** Eliminating the residual requires either porting
  JEOD's DE4xx reader to Rust (~200 lines of Chebyshev evaluation, tracked
  in #27 option A) or linking against cspice, which would contradict the
  pure-Rust scope of this project. The offset is constant in direction
  and bounded — it does not indicate a formula discrepancy.

## SRP thermal-coupling residual

- **Symptom / magnitude:** ~0.034 m position per component over 23 days on
  SIM_3_ORBIT (down from ~3 m when plate temperatures were integrated
  through every RK4 stage); ~83 m on SIM_3_ORBIT_1st_ORDER. Earlier
  Euler-at-dt=1 s variant was 24.7 m and documented in #13.
- **Where observed:** `crates/jeod_runner/tests/tier3_sim_srp.rs:343` and
  `tier3_sim_srp_1st_order.rs:279`; closure rationale in
  simnaut/bevy_jeod#13; remaining convergence path tracked in #114
  (tasks 5.33 / 5.34, DynManager multi-integrable-object scheduling).
- **Root cause:** On standard SIM_3_ORBIT, JEOD's `radiation.sm`
  scheduled-class path computes `temp_dot()` once per integration step
  and holds that derivative constant across the RK4 stages; our Rust
  port intentionally matches that behavior. The remaining ~0.034 m
  residual is therefore not a per-stage-vs-per-step thermal scheduling
  mismatch on this path, but the accumulated effect of smaller
  irreducible numerical differences in the coupled SRP/thermal update
  pipeline. By contrast, configs that use derivative-class / first-order
  thermal integration (SIM_3_ORBIT_1st_ORDER) still expose the larger
  scheduling-coupling gap discussed in #114, where JEOD advances plate
  temperatures as part of the coupled ODE state via
  `ThermalIntegrableObject`.
- **Why irreducible:** For SIM_3_ORBIT, the port already matches JEOD's
  scheduled-per-step thermal update, so the remaining ~0.03 m residual
  meets the <5 m Phase 5 budget and is accepted as numerical noise
  rather than a missing integrator feature. The DynManager
  multi-integrable-object work in #114 remains relevant to derivative-
  class / first-order SRP configurations, where matching JEOD requires
  adding plate temperatures to the RK4 state vector and scheduling them
  alongside orbital state.

## Host libm sin/cos on drag velocity schedule

- **Symptom / magnitude:** velocity-schedule reconstruction agrees with
  JEOD's logged inertial velocity to <1.1e-12 m/s — a few ULPs of
  |v|≈7500 m/s.
- **Where observed:**
  `crates/jeod_runner/tests/tier3_sim_drag_ver.rs:118` (tolerance
  rationale) and line 124 (assertion).
- **Root cause:** Both sides reduce the planet-fixed-to-inertial rotation
  to `libm::sin`/`cos` of the same f64 argument. Differences are limited
  to 1 ULP of the transcendental output times the 7 500 m/s velocity
  magnitude.
- **Why irreducible:** The test's purpose is to surface any cross-host
  libm drift as a dedicated, low-noise assertion before it pollutes the
  drag-force tolerance below. Tightening below a few ULP of |v| would
  require a shared transcendental library between Rust and JEOD's host.

## Geodetic longitude near the poles

- **Symptom / magnitude:** ~3.3e-5 rad longitude error on polar orbits
  (RUN_ell_polar, RUN_sph_polar) vs ~6.5e-8 rad on the inclined RUN_sph_inc.
  Altitude and latitude remain well-behaved at ~2e-4 m and ~1e-8 rad.
- **Where observed:**
  `crates/jeod_runner/tests/tier3_sim_ned_edge.rs:244` (polar) and
  `:268` (inclined) — tolerance comments at lines 246-250; also flagged in
  `CLAUDE.md` "Common Pitfalls".
- **Root cause:** At latitude ±90° every meridian converges on the pole,
  so `atan2(y, x)` becomes hypersensitive: at 89.8° latitude the
  sensitivity is ~3.7e-6 rad per metre of horizontal position error.
  Our sub-millimetre position drift against JEOD (see trajectory entry
  above) therefore amplifies to tens of microradians in longitude.
- **Why irreducible:** Longitude is geometrically undefined at the pole;
  both implementations return valid but numerically unstable values.
  Widening the polar-longitude tolerance is the correct response —
  tightening it would constrain a quantity that is not physically
  observable at that latitude.
