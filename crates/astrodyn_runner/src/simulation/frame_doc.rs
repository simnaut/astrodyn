//! Frame-document export / apply / series recording for [`Simulation`]
//! (issue #663; feature `frame-doc`).
//!
//! The runner is the producing host: it owns the origin knowledge the
//! tree does not carry (which nodes project the authoritative body store,
//! which are model evaluations, which are caller-injected) and the time
//! anchor a restoring host must match.
//!
//! ## Restore model
//!
//! [`Simulation::apply_frame_document`] restores **frame-layer state**
//! into a freshly built simulation that was configured the same way as
//! the producer *as of the snapshot* — for a snapshot taken after a
//! frame switch, that means the post-switch configuration (the body's
//! `integ_source` set to the switch target, the switch inactive or
//! absent, gravity-control `differential` flags pre-flipped). Apply
//! **checks** each record's declared parent identity against the rebuilt
//! tree and fails loudly on mismatch; it never reparents and never
//! touches gravity controls.
//!
//! Scope (documented limits, not silent ones): continuation is bit-exact
//! for single-step integrators (RK4/RKF45) — multistep history
//! (Gauss-Jackson, ABM4, LSODE) is body-store state outside the frame
//! document. The Earth-RNP refresh cadence must be 0 (the default): a
//! non-zero cadence's `EarthRnpCache` is `Simulation`-internal state the
//! document does not carry. Contact-pair pending initial impulses are
//! likewise not restored.

use std::collections::HashMap;

use astrodyn::frame_doc::{
    Conventions, DocHeader, FrameDocument, FrameRecord, FrameSeries, FrameUid, Origin,
    SeriesBuilder, SCHEMA_VERSION,
};
use astrodyn::frame_doc_io::{export_tree, record_epoch, record_state};
use astrodyn::typed_bridge::{rot_raw_to_self_ref, rot_typed_to_raw, trans_raw_to_typed};
use astrodyn::{
    FrameId, IntegrationFrame, JeodQuat, RotationModel, RotationalState, TranslationalState,
};

use super::Simulation;

impl Simulation {
    /// The document header for this simulation's current instant.
    fn doc_header(&self) -> DocHeader {
        DocHeader {
            schema_version: SCHEMA_VERSION,
            conventions: Conventions::current(),
            simtime: self.time.simtime,
            tai_tjt_at_epoch: self.time.tai_tjt_at_epoch,
        }
    }

    /// Classify a frame node's [`Origin`] from the runner's registration
    /// knowledge (RFS-603): body frames project the authoritative body
    /// store (**integrated**, with the store's rotational half as
    /// payload); source frames are model evaluations (**derived**) when a
    /// rotation model or ephemeris mapping drives them, else
    /// caller-supplied (**injected**); the root is injected by
    /// construction.
    ///
    /// # Panics
    /// Panics on a frame this simulation did not register — every
    /// production node is created by `Simulation::new` / `add_source*` /
    /// `add_body`, so an unclassifiable node is a construction-path bug.
    fn frame_origin_classification(&self, id: FrameId) -> Origin {
        if id == self.root_frame_id {
            return Origin::Injected;
        }
        for (i, sf) in self.source_frame_ids.iter().enumerate() {
            if sf.inertial == id {
                return match &self.source_ephem_bodies[i] {
                    Some((target, observer)) => Origin::Derived {
                        model: format!("DE4xx:{target:?}/{observer:?}"),
                    },
                    None => Origin::Injected,
                };
            }
            if sf.pfix == Some(id) {
                return match self.gravity_data[i].rotation_model {
                    // A pfix frame with no rotation model exists because
                    // the caller supplied a fixed `t_inertial_pfix`.
                    RotationModel::None => Origin::Injected,
                    model => Origin::Derived {
                        model: format!("{model:?}"),
                    },
                };
            }
        }
        for body in &self.bodies {
            if body.body_frame_id == id {
                let (attitude_quat, ang_vel_body) = match &body.rot {
                    Some(rot) => {
                        let raw = rot_typed_to_raw(rot);
                        (Some(raw.quaternion.data), Some(raw.ang_vel_body.to_array()))
                    }
                    None => (None, None),
                };
                return Origin::Integrated {
                    attitude_quat,
                    ang_vel_body,
                };
            }
        }
        panic!(
            "frame_origin_classification: frame {id} (`{}`) was not registered by this \
             simulation's construction paths (root / add_source* / add_body) — cannot \
             classify its origin for export.",
            self.frame_tree.get(id).name
        );
    }

    /// Serialize the simulation's frame layer into a snapshot
    /// [`FrameDocument`] (issue #663). Every record carries its parent
    /// identity, epoch, origin, and canonical rotation representation;
    /// see `astrodyn::frame_doc_io::export_tree` for the canonicity and
    /// unstamped-node rules.
    pub fn export_frame_document(&self) -> FrameDocument {
        export_tree(&self.frame_tree, self.doc_header(), |id| {
            self.frame_origin_classification(id)
        })
    }

    /// Restore a snapshot document into this freshly built simulation —
    /// see the module docs for the restore model and scope limits.
    ///
    /// Advances this simulation's clock to the document's `simtime` in a
    /// single exact step, writes every record's node state and epoch
    /// **verbatim**, writes the authoritative body store from each
    /// `Integrated` record's payload, and syncs static source velocities.
    /// Closes the registration window (`has_stepped`): the restored
    /// simulation is semantically mid-run.
    ///
    /// # Panics
    /// - this simulation has already stepped or its clock is not at zero;
    /// - the document's time epoch differs from this simulation's (a
    ///   restored run under a different epoch would silently reinterpret
    ///   every derived time scale);
    /// - a record's identity is missing from this simulation's tree, or
    ///   its declared parent differs from the tree's actual parent
    ///   (rebuild the configuration to the document's topology — e.g.
    ///   the post-frame-switch `integ_source`);
    /// - an `Integrated` record's rotational payload disagrees with the
    ///   body's degrees of freedom (6-DOF body without a payload, or
    ///   payload on a 3-DOF body).
    pub fn apply_frame_document(&mut self, doc: &FrameDocument) {
        assert!(
            !self.has_stepped,
            "apply_frame_document: this simulation has already stepped — restore \
             targets a freshly built simulation configured like the producer at \
             the snapshot instant."
        );
        assert!(
            self.time.simtime.to_bits() == 0.0_f64.to_bits(),
            "apply_frame_document: simulation clock is at simtime {} (expected 0) — \
             restore advances the clock itself; do not pre-advance.",
            self.time.simtime
        );
        assert!(
            self.time.tai_tjt_at_epoch.to_bits() == doc.header.tai_tjt_at_epoch.to_bits(),
            "apply_frame_document: time-epoch mismatch — the document was produced at \
             tai_tjt_at_epoch {}, this simulation is configured at {}. Derived time \
             scales (TDB, GMST) are functions of the epoch; rebuild the simulation \
             with the producer's epoch.",
            doc.header.tai_tjt_at_epoch,
            self.time.tai_tjt_at_epoch
        );
        // One exact addition from zero: bit-identical to the producer's
        // accumulated simtime, and every derived scale is a pure function
        // of it (constant scale factor).
        self.time.advance(doc.header.simtime);

        // body_frame_id → body index, for routing Integrated payloads.
        let body_of_frame: HashMap<FrameId, usize> = self
            .bodies
            .iter()
            .enumerate()
            .map(|(idx, b)| (b.body_frame_id, idx))
            .collect();

        for (pos, rec) in doc.records.iter().enumerate() {
            let uid = &doc.uids[rec.uid_index as usize];
            let fid = self.frame_tree.find(uid).unwrap_or_else(|| {
                panic!(
                    "apply_frame_document: record {pos} (`{}`) has identity `{uid}` but \
                     this simulation's tree has no such frame — rebuild the scenario \
                     configuration to match the document's frame population.",
                    rec.name
                )
            });
            // JEOD_INV: RF.02 — consumers check folded topology against
            // each record's declared parent; a mismatch is a loud
            // inconsistency, never a silent reinterpretation (and apply
            // never reparents).
            let actual_parent = self.frame_tree.parent(fid).map(|pid| {
                self.frame_tree
                    .get(pid)
                    .uid()
                    .expect("production nodes are stamped (issue #662)")
                    .clone()
            });
            let declared_parent: Option<FrameUid> =
                rec.parent.map(|p| doc.uids[p as usize].clone());
            assert!(
                actual_parent == declared_parent,
                "apply_frame_document: topology mismatch at record {pos} (`{}`): the \
                 document declares parent {declared_parent:?}, but this simulation's \
                 tree has {actual_parent:?}. apply never reparents — rebuild the \
                 configuration to the document's topology (for a post-frame-switch \
                 snapshot: set the body's integ_source to the switch target, \
                 deactivate the switch, and pre-flip the gravity-control \
                 differential flags).",
                rec.name
            );

            // Node state + epoch, verbatim (the non-canonical rotation
            // representation is re-derived exactly as the producer's
            // write site derived it).
            self.frame_tree.get_mut(fid).state = record_state(rec);
            self.frame_tree.set_epoch(fid, record_epoch(rec));

            match &rec.origin {
                Origin::Integrated {
                    attitude_quat,
                    ang_vel_body,
                } => {
                    let idx = *body_of_frame.get(&fid).unwrap_or_else(|| {
                        panic!(
                            "apply_frame_document: record {pos} (`{}`) is Integrated but \
                             frame {fid} is not a body frame in this simulation.",
                            rec.name
                        )
                    });
                    // The store's translational half equals the node's by
                    // dual-write construction; the rotational half is the
                    // payload (the body NODE's rotation is stale by
                    // design — writing the store via the dual-write
                    // setters would desync it from the producer).
                    self.bodies[idx].trans =
                        trans_raw_to_typed::<IntegrationFrame>(&TranslationalState {
                            position: glam::DVec3::from_array(rec.trans.position),
                            velocity: glam::DVec3::from_array(rec.trans.velocity),
                        });
                    match (attitude_quat, ang_vel_body, self.bodies[idx].rot.is_some()) {
                        (Some(q), Some(w), true) => {
                            // The Integrated payload's scalar-first layout
                            // is the document's rotation convention (RF.07),
                            // validated in the header; rot_raw_to_self_ref
                            // re-checks unit norm on lift.
                            self.bodies[idx].rot = Some(rot_raw_to_self_ref(&RotationalState {
                                // allowed: wire-deserialization boundary (RF.07 layout, header-validated)
                                quaternion: JeodQuat::from_array(*q),
                                ang_vel_body: glam::DVec3::from_array(*w),
                            }));
                        }
                        (None, None, false) => {}
                        (payload_q, payload_w, has_rot) => panic!(
                            "apply_frame_document: record {pos} (`{}`) rotational payload \
                             (attitude: {}, ang_vel: {}) disagrees with the body's \
                             degrees of freedom (6-DOF: {has_rot}) — the rebuilt \
                             VehicleConfig must match the producer's.",
                            rec.name,
                            payload_q.is_some(),
                            payload_w.is_some(),
                        ),
                    }
                }
                Origin::Derived { .. } | Origin::Injected => {
                    // Source frames: keep the per-source velocity cache in
                    // sync with the restored node (mirrors
                    // `set_source_state`). Load-bearing only for static
                    // sources — stage 2 recomputes model-driven sources
                    // from the restored time before gravity reads them.
                    for (i, sf) in self.source_frame_ids.iter().enumerate() {
                        if sf.inertial == fid {
                            self.gravity_data[i].velocity =
                                glam::DVec3::from_array(rec.trans.velocity);
                        }
                    }
                }
            }
        }

        // The restored simulation is semantically mid-run: close the
        // registration window so late add_body / contact registration
        // fails loudly, exactly as it would on the producer.
        self.has_stepped = true;
    }
}

/// Per-step frame-series recorder (replay v1, issue #663): call
/// [`record`](Self::record) after each step (and once before the first
/// step for the t₀ keyframe); segments close automatically at every
/// topology change — the frame-switch reparent boundary doubles as a
/// seek keyframe.
#[derive(Debug)]
pub struct FrameSeriesRecorder {
    builder: SeriesBuilder,
}

impl FrameSeriesRecorder {
    /// Start recording from `sim`'s current instant (the header's
    /// `simtime` anchors the series at the first recorded epoch).
    pub fn new(sim: &Simulation) -> Self {
        let doc = sim.export_frame_document();
        Self {
            builder: SeriesBuilder::new(doc.header, doc.uids),
        }
    }

    /// Record the simulation's current frame state as one epoch row.
    ///
    /// # Panics
    /// Panics if the frame population changed since [`Self::new`] —
    /// replay v1 records a fixed population (topology may change;
    /// membership may not). The runner's registration window already
    /// rejects mid-run additions, so this only fires on misuse.
    pub fn record(&mut self, sim: &Simulation) {
        let doc = sim.export_frame_document();
        assert!(
            doc.uids == self.builder.uids(),
            "FrameSeriesRecorder::record: the frame population changed since \
             recording began — replay v1 records a fixed population."
        );
        self.record_rows(sim.time.simtime, doc.records);
    }

    fn record_rows(&mut self, simtime: f64, rows: Vec<FrameRecord>) {
        self.builder.push_epoch(simtime, rows);
    }

    /// Finish recording and return the series.
    pub fn finish(self) -> FrameSeries {
        self.builder.finish()
    }
}
