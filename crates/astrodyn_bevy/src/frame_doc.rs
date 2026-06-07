//! Frame-document export / apply for the ECS frame store (issue #664;
//! feature `frame-doc`) — the Bevy mirror of
//! `astrodyn_runner::Simulation::{export,apply}_frame_document`.
//!
//! The ECS hierarchy **is** the Bevy frame store (no arena), so export
//! walks the frame *entities* (every entity carrying
//! [`FrameUidC`] + frame state) and apply writes entity
//! components — the runner's document module is the *semantic* blueprint
//! (header, origin taxonomy, pre-step asserts, single time advance,
//! verbatim node state, Integrated body-store payload, never-reparent
//! topology checks), the mechanism differs.
//!
//! ## Restore model
//!
//! [`apply_frame_document`] restores frame-layer state into a freshly
//! built App that was configured the same way as the producer *as of the
//! snapshot* — for a snapshot taken after a frame switch, that means the
//! post-switch configuration. Apply **checks** each record's declared
//! parent identity against the rebuilt `ChildOf` hierarchy and fails
//! loudly on mismatch; it never reparents.
//!
//! **Sequencing precondition**: apply must run after the App's `Startup`
//! registration (so the frame entities and [`FrameUidIndexR`] exist) and
//! before the first `FixedUpdate` tick (the simulation clock must still
//! be at zero — asserted). Drive `app.update()` once, then call apply,
//! then step.
//!
//! Scope limits mirror the runner's: single-step integrators (multistep
//! history is body-store state outside the frame document); single-`<P>`
//! apply (the same central-planet shape as
//! [`populate_app`](crate::SimulationBuilderBevyExt::populate_app)).

use astrodyn::frame_doc::{
    CanonicalRotation, Conventions, DocHeader, FrameDocument, FrameRecord, Origin, TransRecord,
    SCHEMA_VERSION,
};
use astrodyn::frame_doc_io::{record_epoch, record_state};
use astrodyn::Planet;
use bevy::prelude::*;
use std::collections::HashMap;

use crate::components::*;
use crate::systems::FrameUidIndexR;
use crate::{RootFrameEntityR, SimulationTimeR};

/// Serialize the App's frame layer into a snapshot
/// [`FrameDocument`]. Every record carries its parent identity, epoch,
/// origin, and canonical rotation representation; record order is the
/// root followed by the remaining frame entities in query order —
/// **order is not part of the document contract** (consumers resolve by
/// uid).
///
/// Origins mirror the runner's classifier: the root and static sources
/// are `Injected`; ephemeris-driven source frames and rotating pfix
/// frames are `Derived` (with the runner-identical model strings);
/// body frames are `Integrated` with the **body entity's**
/// [`RotationalStateC`] as payload (the body store is truth; the frame
/// entity's rotation is stale by design, exactly as in the runner).
///
/// # Panics
/// - a frame entity is missing [`FrameEpochC`] or its parent is missing
///   [`FrameUidC`] (registration stamps both — issue #664);
/// - a frame entity is not classifiable (not the root, not a registered
///   source's inertial/pfix frame, not a registered body's frame);
/// - the produced document fails validation (non-finite state must not
///   be laundered into "data").
pub fn export_frame_document(world: &mut World) -> FrameDocument {
    let time = world.resource::<SimulationTimeR>();
    let header = DocHeader {
        schema_version: SCHEMA_VERSION,
        conventions: Conventions::current(),
        simtime: time.simtime,
        tai_tjt_at_epoch: time.tai_tjt_at_epoch,
    };
    let root = world.resource::<RootFrameEntityR>().0;

    // ── Origin classification maps (frame entity → origin) ──
    // Mirror `Simulation::frame_origin_classification`: host knowledge
    // lives on the source/body entities; build reverse maps through the
    // FrameEntityC / PfixFrameEntityC handles.
    let mut origins: HashMap<Entity, Origin> = HashMap::new();
    origins.insert(root, Origin::Injected);

    let mut sources = world.query_filtered::<(
        &FrameEntityC,
        Option<&PfixFrameEntityC>,
        Option<&EphemerisBodyC>,
        Option<&RotationModelC>,
    ), With<GravitySourceC>>();
    for (fe, pfix_fe, ephem, model) in sources.iter(world) {
        let inertial_origin = match ephem {
            Some(e) => Origin::Derived {
                model: format!("DE4xx:{:?}/{:?}", e.target, e.observer),
            },
            None => Origin::Injected,
        };
        origins.insert(fe.0, inertial_origin);
        if let Some(pfix) = pfix_fe {
            // Same default the rotation/registration systems apply: a
            // source carrying `PlanetFixedRotationC` without an explicit
            // `RotationModelC` rotates under `EarthRNP`. An explicit
            // `RotationModel::None` means the pfix orientation is a
            // caller-supplied fixed transform (Injected — the runner's
            // taxonomy for `t_inertial_pfix` without a model).
            let effective = model.map_or(astrodyn::RotationModel::EarthRNP, |m| m.0);
            let pfix_origin = match effective {
                astrodyn::RotationModel::None => Origin::Injected,
                m => Origin::Derived {
                    model: format!("{m:?}"),
                },
            };
            origins.insert(pfix.0, pfix_origin);
        }
    }

    let mut bodies =
        world.query_filtered::<(&FrameEntityC, Option<&RotationalStateC>), With<DynamicsConfigC>>();
    for (fe, rot) in bodies.iter(world) {
        let (attitude_quat, ang_vel_body) = match rot {
            Some(r) => {
                let raw = astrodyn::typed_bridge::rot_typed_to_raw(&r.0);
                (Some(raw.quaternion.data), Some(raw.ang_vel_body.to_array()))
            }
            None => (None, None),
        };
        origins.insert(
            fe.0,
            Origin::Integrated {
                attitude_quat,
                ang_vel_body,
            },
        );
    }

    // ── Walk the frame entities (root first; order is not contractual) ──
    let mut frames = world.query::<(
        Entity,
        &FrameUidC,
        &FrameTransC,
        &FrameRotC,
        &FrameAngVelC,
        Option<&FrameEpochC>,
        Option<&Name>,
        Option<&ChildOf>,
        Has<PlanetFixedFrameMarker>,
    )>();
    let mut ordered: Vec<Entity> = vec![root];
    for (entity, ..) in frames.iter(world) {
        if entity != root {
            ordered.push(entity);
        }
    }

    let mut uids = Vec::with_capacity(ordered.len());
    let mut records = Vec::with_capacity(ordered.len());
    for (idx, &entity) in ordered.iter().enumerate() {
        let (_, uid, trans, rot, ang_vel, epoch, name, child_of, is_pfix) = frames
            .get(world, entity)
            .expect("ordered entities carry frame components");
        let epoch = epoch.unwrap_or_else(|| {
            panic!(
                "export_frame_document: frame entity {entity:?} has no FrameEpochC — \
                 registration stamps every frame entity (issue #664); do not strip it."
            )
        });
        let parent = child_of.map(|c| {
            let parent_entity = c.parent();
            let pos = ordered
                .iter()
                .position(|&e| e == parent_entity)
                .unwrap_or_else(|| {
                    panic!(
                        "export_frame_document: frame entity {entity:?}'s parent \
                         {parent_entity:?} carries no FrameUidC / frame state — the \
                         ChildOf hierarchy must contain only stamped frame entities."
                    )
                });
            u32::try_from(pos).expect("frame population exceeds u32 — unsupported document size")
        });
        // Canonicity is marker-driven, mirroring the runner's
        // class-driven rule: pfix frames are matrix-canonical
        // (planet_fixed_rotation_system writes the model matrix
        // verbatim and derives q), everything else quat-canonical
        // (RF.04).
        let rotation = if is_pfix {
            CanonicalRotation::Matrix(rot.t_parent_this.to_cols_array_2d())
        } else {
            CanonicalRotation::Quat(rot.q_parent_this.data)
        };
        let origin = origins.remove(&entity).unwrap_or_else(|| {
            panic!(
                "export_frame_document: frame entity {entity:?} (`{}`) is not the \
                 root, a registered source's inertial/pfix frame, or a registered \
                 body's frame — cannot classify its origin for export.",
                name.map(|n| n.as_str()).unwrap_or("<unnamed>")
            )
        });
        uids.push(uid.0.clone());
        records.push(FrameRecord {
            name: name
                .map(|n| n.as_str().to_string())
                .unwrap_or_else(|| format!("{entity:?}")),
            uid_index: u32::try_from(idx)
                .expect("frame population exceeds u32 — unsupported document size"),
            parent,
            epoch: Some(epoch.0.as_seconds()),
            trans: TransRecord {
                position: trans.position.to_array(),
                velocity: trans.velocity.to_array(),
            },
            rotation,
            ang_vel_this: ang_vel.0.to_array(),
            origin,
        });
    }

    let doc = FrameDocument {
        header,
        uids,
        records,
    };
    doc.validate()
        .unwrap_or_else(|err| panic!("export_frame_document: produced an invalid document: {err}"));
    doc
}

/// Restore a snapshot document into this freshly built App — see the
/// module docs for the restore model and sequencing precondition.
///
/// Advances the App's clock to the document's `simtime` in a single
/// exact step, writes every record's frame-entity state and epoch
/// **verbatim**, writes the body store
/// ([`TranslationalStateC<P>`] / [`RotationalStateC`]) from each
/// `Integrated` record, and syncs static source components.
///
/// # Panics
/// - the App has already ticked (clock not at zero) or its time epoch
///   differs from the document's;
/// - a record's identity is missing from [`FrameUidIndexR`], or its
///   declared parent differs from the `ChildOf` hierarchy (rebuild the
///   configuration to the document's topology — apply never reparents);
/// - an `Integrated` record's body entity does not carry
///   [`TranslationalStateC<P>`] (single-`<P>` apply, matching
///   `populate_app`'s central-planet shape) or its rotational payload
///   disagrees with the body's degrees of freedom.
pub fn apply_frame_document<P: Planet>(world: &mut World, doc: &FrameDocument) {
    {
        let mut time = world.resource_mut::<SimulationTimeR>();
        assert!(
            time.simtime.to_bits() == 0.0_f64.to_bits(),
            "apply_frame_document: simulation clock is at simtime {} (expected 0) — \
             restore targets a freshly built App before its first FixedUpdate tick.",
            time.simtime
        );
        assert!(
            time.tai_tjt_at_epoch.to_bits() == doc.header.tai_tjt_at_epoch.to_bits(),
            "apply_frame_document: time-epoch mismatch — the document was produced at \
             tai_tjt_at_epoch {}, this App is configured at {}. Derived time scales \
             (TDB, GMST) are functions of the epoch; rebuild the App with the \
             producer's epoch.",
            doc.header.tai_tjt_at_epoch,
            time.tai_tjt_at_epoch
        );
        // One exact addition from zero — bit-identical to the producer's
        // accumulated clock (mirrors the runner's apply).
        time.advance(doc.header.simtime);
    }

    // frame entity → owning body/source entity reverse maps for the
    // store writes.
    let mut body_of_frame: HashMap<Entity, Entity> = HashMap::new();
    let mut bodies = world.query_filtered::<(Entity, &FrameEntityC), With<DynamicsConfigC>>();
    for (entity, fe) in bodies.iter(world) {
        body_of_frame.insert(fe.0, entity);
    }
    let mut source_of_frame: HashMap<Entity, Entity> = HashMap::new();
    let mut sources = world.query_filtered::<(Entity, &FrameEntityC), With<GravitySourceC>>();
    for (entity, fe) in sources.iter(world) {
        source_of_frame.insert(fe.0, entity);
    }

    for (pos, rec) in doc.records.iter().enumerate() {
        let uid = &doc.uids[rec.uid_index as usize];
        let entity = *world
            .resource::<FrameUidIndexR>()
            .0
            .get(uid)
            .unwrap_or_else(|| {
                panic!(
                    "apply_frame_document: record {pos} (`{}`) has identity `{uid}` but \
                     this App's frame store has no such frame — rebuild the scenario \
                     configuration to match the document's frame population (and drive \
                     app.update() once so registration runs before apply).",
                    rec.name
                )
            });

        // JEOD_INV: RF.02 — consumers check folded topology against each
        // record's declared parent; a mismatch is a loud inconsistency,
        // never a silent reinterpretation (and apply never reparents).
        let actual_parent = world.entity(entity).get::<ChildOf>().map(|c| {
            world
                .entity(c.parent())
                .get::<FrameUidC>()
                .unwrap_or_else(|| {
                    panic!(
                        "apply_frame_document: frame entity {entity:?}'s parent carries \
                         no FrameUidC — frame entities are stamped at registration."
                    )
                })
                .0
                .clone()
        });
        let declared_parent = rec.parent.map(|p| doc.uids[p as usize].clone());
        assert!(
            actual_parent == declared_parent,
            "apply_frame_document: topology mismatch at record {pos} (`{}`): the \
             document declares parent {declared_parent:?}, but this App's hierarchy \
             has {actual_parent:?}. apply never reparents — rebuild the configuration \
             to the document's topology (for a post-frame-switch snapshot: wire the \
             body's IntegSourceC to the switch target and pre-flip the gravity \
             controls).",
            rec.name
        );

        // Frame-entity state + epoch, verbatim (the non-canonical
        // rotation representation is re-derived exactly as the
        // producer's write site derived it).
        let state = record_state(rec);
        let epoch = record_epoch(rec).unwrap_or_else(|| {
            panic!(
                "apply_frame_document: record {pos} (`{}`) carries no epoch — \
                 producer trees stamp every node (issue #662/#664).",
                rec.name
            )
        });
        let mut entity_mut = world.entity_mut(entity);
        *entity_mut
            .get_mut::<FrameTransC>()
            .expect("frame entity has FrameTransC") = FrameTransC {
            position: state.trans.position,
            velocity: state.trans.velocity,
        };
        *entity_mut
            .get_mut::<FrameRotC>()
            .expect("frame entity has FrameRotC") = FrameRotC {
            q_parent_this: state.rot.q_parent_this,
            t_parent_this: state.rot.t_parent_this,
        };
        *entity_mut
            .get_mut::<FrameAngVelC>()
            .expect("frame entity has FrameAngVelC") = FrameAngVelC(state.rot.ang_vel_this);
        *entity_mut
            .get_mut::<FrameEpochC>()
            .expect("frame entity has FrameEpochC") = FrameEpochC(epoch);

        match &rec.origin {
            Origin::Integrated {
                attitude_quat,
                ang_vel_body,
            } => {
                let body = *body_of_frame.get(&entity).unwrap_or_else(|| {
                    panic!(
                        "apply_frame_document: record {pos} (`{}`) is Integrated but \
                         frame entity {entity:?} is not a registered body's frame.",
                        rec.name
                    )
                });
                let mut body_mut = world.entity_mut(body);
                // The store's translational half equals the frame
                // entity's by dual-write construction; the rotational
                // half is the payload (the frame entity's rotation is
                // stale by design — mirroring the runner's apply).
                let mut trans = body_mut
                    .get_mut::<TranslationalStateC<P>>()
                    .unwrap_or_else(|| {
                        panic!(
                            "apply_frame_document: body {body:?} (record `{}`) has no \
                             TranslationalStateC<{}> — apply is single-planet, matching \
                             populate_app's central-planet shape; call \
                             apply_frame_document::<P> with the App's central planet.",
                            rec.name,
                            core::any::type_name::<P>()
                        )
                    });
                // allowed: wire-deserialization boundary — record values are
                // root/integ-frame coordinates per the document conventions,
                // validated on load.
                trans.0 = astrodyn::typed_bridge::trans_raw_to_planet::<P>(
                    &astrodyn::TranslationalState {
                        position: state.trans.position,
                        velocity: state.trans.velocity,
                    },
                );
                let has_rot = body_mut.get::<RotationalStateC>().is_some();
                match (attitude_quat, ang_vel_body, has_rot) {
                    (Some(q), Some(w), true) => {
                        let mut rot = body_mut
                            .get_mut::<RotationalStateC>()
                            .expect("checked has_rot above");
                        rot.0 = astrodyn::typed_bridge::rot_raw_to_self_ref(
                            &astrodyn::RotationalState {
                                // allowed: wire-deserialization boundary (RF.07
                                // layout, header-validated)
                                quaternion: astrodyn::JeodQuat::from_array(*q),
                                ang_vel_body: glam::DVec3::from_array(*w),
                            },
                        );
                    }
                    (None, None, false) => {}
                    (payload_q, payload_w, has_rot) => panic!(
                        "apply_frame_document: record {pos} (`{}`) rotational payload \
                         (attitude: {}, ang_vel: {}) disagrees with the body's degrees \
                         of freedom (6-DOF: {has_rot}) — the rebuilt configuration must \
                         match the producer's.",
                        rec.name,
                        payload_q.is_some(),
                        payload_w.is_some(),
                    ),
                }
            }
            Origin::Derived { .. } | Origin::Injected => {
                // Source frames: keep the source-entity components in
                // sync with the restored frame state (mirrors the
                // runner syncing gravity_data.velocity). Load-bearing
                // only for static sources — the ephemeris/rotation
                // systems recompute model-driven sources from the
                // restored time on the first tick.
                if let Some(&source) = source_of_frame.get(&entity) {
                    let mut source_mut = world.entity_mut(source);
                    if let Some(mut pos_c) = source_mut.get_mut::<SourceInertialPositionC>() {
                        // allowed: wire-deserialization boundary — document
                        // conventions (root-inertial, SI) validated on load.
                        pos_c.0 = astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                            state.trans.position,
                        );
                    }
                    if let Some(mut vel_c) = source_mut.get_mut::<SourceInertialVelocityC>() {
                        // allowed: wire-deserialization boundary (as above)
                        vel_c.0 = astrodyn::Velocity::<astrodyn::RootInertial>::from_raw_si(
                            state.trans.velocity,
                        );
                    }
                    // `TranslationalStateC<P>` on the source entity is what
                    // SRP / lighting / interaction queries read — leaving it
                    // stale would diverge a restored App immediately for
                    // static sources (and until the first ephemeris tick for
                    // model-driven ones).
                    if let Some(mut trans_c) = source_mut.get_mut::<TranslationalStateC<P>>() {
                        // allowed: wire-deserialization boundary (as above);
                        // named-method opt-in lift matching spawn_source.
                        trans_c.0 = astrodyn::typed_bridge::trans_raw_to_planet::<P>(
                            &astrodyn::TranslationalState {
                                position: state.trans.position,
                                velocity: state.trans.velocity,
                            },
                        );
                    }
                }
            }
        }
    }
}
