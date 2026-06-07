//! Shared SIM_Apollo `apollo_trajectory` recipe + topology helpers.
//!
//! Factored out of `crates/astrodyn_verif_jeod/tests/tier3_sim_apollo_trajectory.rs`
//! so the runner-vs-JEOD tier3 test and the runner-vs-Bevy parity wrapper
//! (`crates/astrodyn_verif_parity/tests/bevy_parity_apollo_trajectory.rs`)
//! drive the same scenario through their respective runtimes.
//!
//! The apollo mass tree pairs *one integrated body* (`cm`, the Command
//! Module — the launch-stack integration root) with *seven tree-only mass
//! bodies* (`sm`, `lm`, `dm`, `s3`, `s2`, `s1`, `les`) plus fourteen
//! named attachment points and seven `launch_stack` aligned attaches.
//! That topology does not fit `SimulationBuilder`'s declarative shape
//! (the builder only registers integrated bodies), so the recipe is
//! split into:
//!
//! 1. [`apollo_trajectory_builder`] — returns a `SimulationBuilder`
//!    carrying time, sources, ephemeris, atmosphere, and the single
//!    integrated `cm` body (registered in the mass tree). Consumed by
//!    `Simulation::from_builder` and `SimulationBuilderBevyExt::populate_app`
//!    identically.
//!
//! 2. [`setup_apollo_arena`] — given a `&mut MassTree` already
//!    containing the `cm` integrated body, adds the seven tree-only
//!    mass bodies, fourteen named mass points, and seven launch-stack
//!    aligned attaches. Both adapters call this on their own arena
//!    (`Simulation::mass_tree` for the runner; `MassTreeR.0` for Bevy)
//!    after the basic build is done. Mass-id allocation is
//!    deterministic — the same insertion order produces the same
//!    `MassBodyId`s on both sides.
//!
//! 3. [`Event`] / [`EVENTS`] — the eleven SIM_Apollo
//!    `RUN_test/input.py:230..345` mass-tree events scheduled at
//!    integer seconds. [`apply_event`] dispatches each event through
//!    a `&mut dyn SimContext`, calling the subtree-detach /
//!    subtree-attach-aligned methods both runtimes implement.
//!
//! ### JEOD source-defect note
//!
//! `sims/SIM_Apollo/SET_test/RUN_test/input.py` calls
//! `set_vehicle_grav_controls()` only on `les_dyn` and never on
//! `cm_dyn` (the integration root after launch_stack assembly). As
//! shipped, JEOD's recorded trajectory is therefore essentially
//! gravity-free. The Docker reference-regen wrapper
//! (`trick/generate_references.sh:run_apollo_group`) injects the
//! missing `set_vehicle_grav_controls(cm_dyn)` +
//! `set_vehicle_sv_at_earth(cm_dyn, earth)` calls before the sim
//! runs, restoring the 8x8 GGM05C + Moon/Sun gravity that the
//! per-vehicle data files (`Modified_data/vehicle/grav_controls.py`,
//! `Modified_data/vehicle/sv_at_earth.py`) clearly intend. This
//! recipe therefore wires the *intended* gravity configuration
//! rather than the as-shipped (broken) one.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Apollo recipe step counts and indices fit exactly in f64 mantissa and usize"
)]

use crate::verification::SimContext;
use astrodyn::{
    AtmosphereConfig, AtmosphereModel, GeoIndexType, GravityControl, GravityControls,
    GravityGradient, GravityModel, GravitySource, GravitySourceEntry, JeodQuat, MassBodyId,
    MassProperties, MassTree, MetAtmosphere, RotationalState, SimulationBuilder, SimulationTime,
    TranslationalState, VehicleConfig, EARTH,
};
use astrodyn_runner::RotationModel;
use glam::{DMat3, DVec3};
use std::path::PathBuf;

// ── JEOD source constants ────────────────────────────────────────────

/// `S_define:72` — `#define DYNAMICS 0.02`.
pub const DT: f64 = 0.02;
/// `RUN_test/input.py:350` — `exec_set_terminate_time(12.0)`.
pub const SIM_DURATION_S: f64 = 12.0;

/// `Modified_data/Earth/params.py` — Earth rotation rate.
pub const OMEGA_EARTH: f64 = 7.292_115_146_706_388e-5;

/// `Modified_data/vehicle/sv_at_earth.py` — earth gravity 8x8 degree.
pub const GRAV_DEGREE: usize = 8;
/// `Modified_data/vehicle/sv_at_earth.py` — earth gravity 8x8 order.
pub const GRAV_ORDER: usize = 8;

// ── Unit conversions for JEOD English-unit mass data ─────────────────

/// Pounds-mass → kilograms (exact).
pub const LB_TO_KG: f64 = 0.453_592_37;
/// Feet → meters (exact).
pub const FT_TO_M: f64 = 0.3048;
const LB_FT2_TO_KG_M2: f64 = LB_TO_KG * FT_TO_M * FT_TO_M;

// ── Topology ─────────────────────────────────────────────────────────

/// MassBodyId handles for the eight Apollo bodies, returned by
/// [`setup_apollo_arena`]. Pre_step closures capture this struct and
/// dispatch the eleven scheduled detach / attach events on it.
#[derive(Debug, Clone, Copy)]
pub struct ApolloTopology {
    /// Command Module — the *only* integrated body (the launch stack's
    /// post-assembly tree root and JEOD integration target).
    pub cm: MassBodyId,
    /// Service Module — tree-only.
    pub sm: MassBodyId,
    /// Lunar Module (Ascent) — tree-only.
    pub lm: MassBodyId,
    /// Descent Module — tree-only.
    pub dm: MassBodyId,
    /// Saturn V stage 3 — tree-only.
    pub s3: MassBodyId,
    /// Saturn V stage 2 — tree-only.
    pub s2: MassBodyId,
    /// Saturn V stage 1 — tree-only.
    pub s1: MassBodyId,
    /// Launch Escape System — tree-only.
    pub les: MassBodyId,
}

// ── 180° yaw rotation about Z ────────────────────────────────────────

/// JEOD's `pt_orientation = yaw_180`: `diag(-1, -1, 1)` — used as
/// `t_struct_to_body` on CM / LES / DM / Ascent module (each declares
/// `eigen_angle = 180°` about Z). Identity for SM / S1 / S2 / S3.
pub fn yaw_180() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    )
}

/// Apollo per-body mass properties from `Modified_data/mass/*.py`.
/// `mass_lb` in pounds, `cm_x_ft` in feet (Y/Z = 0), inertia in lb·ft².
/// `t_struct_to_body` per JEOD `pt_orientation`.
pub fn apollo_mass(
    mass_lb: f64,
    cm_x_ft: f64,
    ixx: f64,
    iyy: f64,
    izz: f64,
    t_struct_to_body: DMat3,
) -> MassProperties {
    MassProperties::with_inertia(
        mass_lb * LB_TO_KG,
        DMat3::from_diagonal(DVec3::new(
            ixx * LB_FT2_TO_KG_M2,
            iyy * LB_FT2_TO_KG_M2,
            izz * LB_FT2_TO_KG_M2,
        )),
        DVec3::new(cm_x_ft * FT_TO_M, 0.0, 0.0),
    )
    .with_t_parent_this(t_struct_to_body)
}

// ── Time setup ───────────────────────────────────────────────────────

/// `Modified_data/date_n_time/UTC_16Jul1969.py` — 1969-07-16 13:44:00 UTC,
/// leap_sec_override = 4.2 s, tai_to_ut1_override = 0.0115221 - 4.2.
pub fn apollo_time() -> SimulationTime {
    // JD(1969-07-16 0h UT) = 2440418.5 → TJT = 418.0 (TJT = JD - 2440000.5).
    // 13h44m = 49440 s = 0.572222... days.
    let utc_tjt = 418.0 + (13.0 * 3600.0 + 44.0 * 60.0) / 86_400.0;
    // SIM_Apollo overrides TAI-UTC to 4.2 s instead of the historical
    // value (which differs at this epoch). Hand-roll the conversion so
    // we don't rely on the leap-second table for this date.
    let tai_tjt = utc_tjt + 4.2 / 86_400.0;
    let mut time = SimulationTime::new(tai_tjt, astrodyn::default_leap_second_table());
    // tai_to_ut1_override_val = 0.0115221 - 4.2 = UT1-TAI offset.
    time.set_ut1_tai_offset(0.011_522_1 - 4.2);
    time
}

// ── Reference-CSV initial state ──────────────────────────────────────

/// Apollo CSV row 0: JEOD's logged `core_body` inertial state at t=0,
/// after `launch_stack` assembled the full stack. Used by the recipe
/// to seed the integrated `cm` body's translational + rotational
/// state; `convert_body_trans_core_to_composite` flips the
/// interpretation to `composite_body` once the mass tree is finalized.
#[derive(Debug, Clone, Copy)]
pub struct ApolloCsvRow0 {
    /// Position in Earth.inertial (m).
    pub position: DVec3,
    /// Velocity in Earth.inertial (m/s).
    pub velocity: DVec3,
    /// Inertial-to-body left-quaternion (JEOD scalar-first layout).
    pub quaternion: JeodQuat,
    /// Body-frame angular velocity (rad/s).
    pub ang_vel_body: DVec3,
}

/// Locate the apollo_trajectory.csv fixture relative to the verif_jeod
/// crate root. Both runtime adapters read row 0 here at scenario-build
/// time (no per-tick CSV access).
pub fn apollo_test_data_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is the directory containing the verif_jeod
    // crate's Cargo.toml. The reference CSV lives under
    // `test_data/apollo_trajectory.csv` in the same crate.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

/// Read row 0 of `apollo_trajectory.csv` — the integration root's
/// initial `core_body` inertial state at the SIM_Apollo epoch.
///
/// Reading CSV row 0 (rather than hard-coding constants) keeps the
/// recipe self-consistent with whatever frame the JEOD reference
/// snippet was logging at t=0. `Modified_data/state/sv_leo_lvlh.py`
/// sets the composite_body state at t=0, but the snippet logs core_body;
/// using row 0 absorbs the structure-to-composite offset times the
/// row-0 rotation without re-deriving it here.
pub fn load_apollo_csv_row0() -> ApolloCsvRow0 {
    let csv_path = apollo_test_data_dir().join("apollo_trajectory.csv");
    assert!(
        csv_path.exists(),
        "apollo_trajectory.csv missing at {}. Generate with: cargo xtask regenerate-tier3",
        csv_path.display()
    );
    let content = std::fs::read_to_string(&csv_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", csv_path.display()));
    // Skip the header, take the first data row.
    let first = content
        .lines()
        .nth(1)
        .unwrap_or_else(|| panic!("{} has no data rows", csv_path.display()));
    let fields: Vec<&str> = first.split(',').map(str::trim).collect();
    assert_eq!(
        fields.len(),
        14,
        "{} row 0: expected 14 columns, got {} ({first:?})",
        csv_path.display(),
        fields.len(),
    );
    let parse = |col: usize| -> f64 {
        fields[col].parse::<f64>().unwrap_or_else(|e| {
            panic!(
                "{} row 0: failed to parse column {col} ({:?}): {e}",
                csv_path.display(),
                fields[col]
            )
        })
    };
    ApolloCsvRow0 {
        position: DVec3::new(parse(1), parse(3), parse(5)),
        velocity: DVec3::new(parse(2), parse(4), parse(6)),
        // JEOD scalar-first [q0, q1, q2, q3] — stored with same convention.
        quaternion: JeodQuat::new(parse(7), parse(8), parse(9), parse(10)),
        ang_vel_body: DVec3::new(parse(11), parse(12), parse(13)),
    }
}

// ── SimulationBuilder factory ────────────────────────────────────────

/// Builder shape returned by [`apollo_trajectory_builder`]. The
/// integrated `cm` body's index in `bodies` is always 0, and `cm` is
/// registered in the mass tree under the name `"cm"`.
pub struct ApolloBuilderHandles {
    /// Pre-built `SimulationBuilder` carrying time, sources, ephemeris,
    /// atmosphere, and the single integrated `cm` body.
    pub builder: SimulationBuilder,
    /// Source index of Earth (`central = true`, RNP rotation).
    pub earth_source_idx: usize,
    /// Source index of the Moon (ephemeris-driven point mass).
    pub moon_source_idx: usize,
    /// Source index of the Sun (ephemeris-driven point mass).
    pub sun_source_idx: usize,
    /// Cached row-0 state used downstream to seed the integrated body
    /// and (after the mass tree is assembled) to convert the `cm`
    /// body's `trans` from `core_body` to `composite_body`.
    pub csv_row0: ApolloCsvRow0,
}

/// Build the SIM_Apollo `apollo_trajectory` simulation as a
/// [`SimulationBuilder`]. Materializing the builder produces a
/// runtime that has the integrated CM body but no tree-only mass
/// bodies — those are added next via [`setup_apollo_arena`] against
/// the runtime's mass-tree arena, mirroring JEOD's two-step setup
/// (DynBody init → MassTree augmentation).
pub fn apollo_trajectory_builder() -> ApolloBuilderHandles {
    // Earth: 8x8 GGM05C non-spherical, with the Earth-RNP rotation model
    // so the planet-fixed frame updates each step (matches JEOD's
    // `earth_GGM05C_MET_RNP.sm`).
    let earth_grav = astrodyn::gravity_fixtures::load_ggm05c();
    let mu_moon = astrodyn::gravity_fixtures::load_moon_grail150_mu();
    let mu_sun = astrodyn::gravity_fixtures::load_sun_spherical_mu();

    let mut sb = SimulationBuilder::new(apollo_time(), DT);

    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: earth_grav.mu,
                model: GravityModel::SphericalHarmonics(Box::new(earth_grav)),
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: OMEGA_EARTH,
            central: true,
            marker_only: false,
        },
    );
    let moon = sb.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_moon,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    let sun = sb.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    sb.set_source_ephemeris(
        moon,
        astrodyn::EphemerisBody::Moon,
        astrodyn::EphemerisBody::Earth,
    );
    sb.set_source_ephemeris(
        sun,
        astrodyn::EphemerisBody::Sun,
        astrodyn::EphemerisBody::Earth,
    );

    // Mean solar activity per `Modified_data/Earth/soflx_mean.py`.
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(MetAtmosphere {
                f10: 128.8,
                f10b: 128.8,
                geo_index: 15.7,
                geo_index_type: GeoIndexType::Ap,
            }),
            r_eq: EARTH.shape.r_eq(),
            r_pol: EARTH.shape.r_pol(),
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );

    // The body starts with CM-only mass; the tree-only bodies + launch
    // stack added in `setup_apollo_arena` augment the composite at runtime
    // via `sync_body_mass_from_tree`.
    let cm_only_mass = apollo_mass(12_807.0, 8.7, 157_372.0, 64_624.0, 64_624.0, yaw_180());

    let csv_row0 = load_apollo_csv_row0();

    sb.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: csv_row0.position,
            velocity: csv_row0.velocity,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: csv_row0.quaternion,
                ang_vel_body: csv_row0.ang_vel_body,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&cm_only_mass)),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GRAV_DEGREE,
                    GRAV_ORDER,
                    GravityGradient::Skip,
                ),
                GravityControl::new_third_body(astrodyn::FrameUid::of::<
                    astrodyn::PlanetInertial<astrodyn::Moon>,
                >()),
                GravityControl::new_third_body(astrodyn::FrameUid::of::<
                    astrodyn::PlanetInertial<astrodyn::Sun>,
                >()),
            ],
        },
        ..VehicleConfig::named("sim-apollo-trajectory-0")
    });

    // Register cm in the mass tree under the canonical name "cm" so the
    // arena ID allocated by `from_builder` / `populate_app` is the same
    // on both runtimes (insertion order is deterministic).
    sb.register_in_mass_tree(0, "cm");

    // astrodyn_runner uses the BSP for Moon/Sun ephemeris evaluation each
    // step. SIM_Apollo's 12 s window sits comfortably within DE421
    // coverage.
    let bsp_path = astrodyn::ephemeris_assets::de421_path();
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris missing at {}",
        bsp_path.display()
    );
    let ephemeris =
        astrodyn::Ephemeris::from_bsp(&bsp_path).expect("failed to load DE421 ephemeris");
    sb = sb.ephemeris(ephemeris);

    ApolloBuilderHandles {
        builder: sb,
        earth_source_idx: earth,
        moon_source_idx: moon,
        sun_source_idx: sun,
        csv_row0,
    }
}

// ── Mass-tree arena setup ────────────────────────────────────────────

/// Given a `&mut MassTree` that already contains the integrated `cm`
/// body, augment it with the seven tree-only mass bodies, fourteen
/// named mass points (per `Modified_data/attach/*.py`), and seven
/// launch_stack aligned attaches.
///
/// `cm` is the integrated body's `MassBodyId` (allocated by
/// `Simulation::from_builder` or `SimulationBuilderBevyExt::populate_app`
/// from `register_in_mass_tree(0, "cm")`). Both runtimes pass their
/// own arena's matching id here; the tree-only bodies are then
/// allocated in deterministic order so both arenas end up holding
/// the same `MassBodyId`s for every node.
///
/// Per `Modified_data/mass/*.py`: SM, S1, S2, S3 use identity
/// struct→body rotation; LM (Ascent), DM, LES use yaw_180.
pub fn setup_apollo_arena(tree: &mut MassTree, cm: MassBodyId) -> ApolloTopology {
    // Allocate the seven tree-only bodies in the same order the
    // pre-refactor inline setup used. Both arenas share the
    // post-`cm` allocation sequence, so their `MassBodyId`s match
    // index-for-index.
    let sm = tree.add_body(
        "sm".into(),
        apollo_mass(
            54_064.0,
            12.3,
            1_107_231.0,
            1_235_227.0,
            1_235_227.0,
            DMat3::IDENTITY,
        ),
    );
    let lm = tree.add_body(
        "lm".into(),
        apollo_mass(10_582.0, 5.45, 259_259.0, 155_822.0, 155_822.0, yaw_180()),
    );
    let dm = tree.add_body(
        "dm".into(),
        apollo_mass(25_640.0, 5.0, 628_180.0, 367_506.0, 367_506.0, yaw_180()),
    );
    let s3 = tree.add_body(
        "s3".into(),
        apollo_mass(
            274_171.0,
            30.65,
            16_138_048.0,
            29_532_558.0,
            29_532_558.0,
            DMat3::IDENTITY,
        ),
    );
    let s2 = tree.add_body(
        "s2".into(),
        apollo_mass(
            1_083_480.0,
            40.75,
            147_488_715.0,
            223_676_545.0,
            223_676_545.0,
            DMat3::IDENTITY,
        ),
    );
    let s1 = tree.add_body(
        "s1".into(),
        apollo_mass(
            5_031_023.0,
            69.0,
            684_848_006.0,
            2_338_482_378.0,
            2_338_482_378.0,
            DMat3::IDENTITY,
        ),
    );
    let les = tree.add_body(
        "les".into(),
        apollo_mass(9_200.0, 16.25, 5_566.0, 205_231.0, 205_231.0, yaw_180()),
    );

    // CM points
    tree.add_mass_point(
        cm,
        "SM interface",
        DVec3::new(11.6 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    tree.add_mass_point(
        cm,
        "CM docking port",
        DVec3::new(4.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    // SM points
    tree.add_mass_point(
        sm,
        "Stage 3 interface",
        DVec3::new(-20.9 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    tree.add_mass_point(
        sm,
        "CM interface",
        DVec3::new(24.6 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // LM points
    tree.add_mass_point(
        lm,
        "LM docking port",
        DVec3::new(10.9 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    tree.add_mass_point(lm, "Descent Module interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        lm,
        "Stage 3 interface",
        DVec3::new(-10.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    // DM points
    tree.add_mass_point(dm, "Ascent Module interface", DVec3::ZERO, DMat3::IDENTITY);
    tree.add_mass_point(
        dm,
        "Stage 3 interface",
        DVec3::new(-10.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    // S3 points
    tree.add_mass_point(s3, "Stage 2 interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        s3,
        "LEM/SM/CM interface",
        DVec3::new(61.3 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // S2 points
    tree.add_mass_point(s2, "Stage 1 interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        s2,
        "Stage 3 interface",
        DVec3::new(81.5 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // S1 points
    tree.add_mass_point(
        s1,
        "Stage 2 interface",
        DVec3::new(138.0 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // LES points
    tree.add_mass_point(les, "CM interface", DVec3::ZERO, yaw_180());

    // Apply the seven `Modified_data/attach/launch_stack.py`
    // attachments. Bottom-up order matters for the composite-mass
    // recomputation that runs after each `attach_aligned`.
    tree.attach_aligned(
        dm,
        "Ascent Module interface",
        lm,
        "Descent Module interface",
    );
    tree.attach_aligned(sm, "CM interface", cm, "SM interface");
    tree.attach_aligned(s3, "LEM/SM/CM interface", sm, "Stage 3 interface");
    tree.attach_aligned(lm, "Stage 3 interface", s3, "LEM/SM/CM interface");
    tree.attach_aligned(s2, "Stage 3 interface", s3, "Stage 2 interface");
    tree.attach_aligned(s1, "Stage 2 interface", s2, "Stage 1 interface");
    tree.attach_aligned(les, "CM interface", cm, "CM docking port");

    ApolloTopology {
        cm,
        sm,
        lm,
        dm,
        s3,
        s2,
        s1,
        les,
    }
}

// ── Mass-tree event schedule (RUN_test/input.py:230..345) ────────────

/// The eleven SIM_Apollo mass-tree events: five staged stage drops
/// (S1 → S2 → LES → S3 → LM), the t=6 SM→CM attach, the t=7..10
/// LM-extract / DM-detach / LM-re-dock / LM-jettison sequence, and
/// the t=11 SM detach.
#[derive(Debug, Clone, Copy)]
pub enum Event {
    /// t=1: drop Stage 1.
    DetachS1,
    /// t=2: drop Stage 2.
    DetachS2,
    /// t=3: jettison Launch Escape System.
    DetachLes,
    /// t=4: drop Stage 3.
    DetachS3,
    /// t=5: separate the LM (transposition + docking sequence start).
    DetachLm,
    /// t=6: dock LM to CM via the named CM-docking-port mass points.
    AttachLmCm,
    /// t=7: undock LM from CM (LM descent begins).
    DetachLm2,
    /// t=8: jettison the descent module (LM-ascent only).
    DetachDm,
    /// t=9: re-dock the LM-ascent stage to the CM.
    AttachLmCm2,
    /// t=10: jettison the LM-ascent stage.
    DetachLm3,
    /// t=11: jettison the Service Module.
    DetachSm,
}

/// The eleven scheduled events with their integer-second firing times,
/// per `RUN_test/input.py:230..345` (`trick.add_read(t, …)` calls).
/// `trick.add_read(t, …)` fires at the END of the cycle ending at `t` —
/// after the integrator has advanced state to `t`. The lockstep driver
/// must step both runtimes up to and including `event_t`, then apply
/// the event before continuing.
pub const EVENTS: &[(f64, Event)] = &[
    (1.0, Event::DetachS1),
    (2.0, Event::DetachS2),
    (3.0, Event::DetachLes),
    (4.0, Event::DetachS3),
    (5.0, Event::DetachLm),
    (6.0, Event::AttachLmCm),
    (7.0, Event::DetachLm2),
    (8.0, Event::DetachDm),
    (9.0, Event::AttachLmCm2),
    (10.0, Event::DetachLm3),
    (11.0, Event::DetachSm),
];

/// Dispatch one [`Event`] through a `&mut dyn SimContext`. Both runtime
/// adapters expose [`SimContext::detach_subtree`] and
/// [`SimContext::attach_subtree_aligned`]; the runner forwards to
/// `Simulation::detach_subtree` / `attach_subtree_aligned`, and the
/// Bevy adapter resolves the `MassBodyId` to its backing mass-only
/// entity and fires an `AttachEvent` / `DetachEvent` (looking up the
/// named mass points in `MassTreeR` to compute offset/rotation per the
/// `MassTree::attach_aligned` chain).
pub fn apply_event(ctx: &mut dyn SimContext, topology: &ApolloTopology, event: Event) {
    match event {
        Event::DetachS1 => ctx.detach_subtree(topology.s1),
        Event::DetachS2 => ctx.detach_subtree(topology.s2),
        Event::DetachLes => ctx.detach_subtree(topology.les),
        Event::DetachS3 => ctx.detach_subtree(topology.s3),
        Event::DetachLm | Event::DetachLm2 | Event::DetachLm3 => {
            ctx.detach_subtree(topology.lm);
        }
        Event::DetachDm => ctx.detach_subtree(topology.dm),
        Event::DetachSm => ctx.detach_subtree(topology.sm),
        Event::AttachLmCm | Event::AttachLmCm2 => {
            ctx.attach_subtree_aligned(
                topology.lm,
                "LM docking port",
                topology.cm,
                "CM docking port",
            );
        }
    }
}
