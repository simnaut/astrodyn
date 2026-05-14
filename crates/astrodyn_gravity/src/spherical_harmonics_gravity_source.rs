// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! [`SphericalHarmonicsData`] — coefficient table and precomputed
//! Gottlieb helper arrays for spherical-harmonics gravity.
//!
//! Ports
//! [`models/environment/gravity/src/spherical_harmonics_gravity_source.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/gravity/src/spherical_harmonics_gravity_source.cc)
//! from JEOD v5.4.0. Coefficients are normalized; the
//! [`crate::spherical_harmonics_calc_nonspherical`] kernel expects the
//! Gottlieb helper arrays to be filled by [`SphericalHarmonicsData::new`].
//!
//! ## Flat triangular storage
//!
//! The six triangular arrays (`cnm`, `snm`, `xi`, `eta`, `zeta`,
//! `upsilon`) are stored as flat `Vec<f64>` indexed by
//! `(n, m) -> n*(n+1)/2 + m`. This packs each row contiguously and
//! removes the row-pointer indirection of a `Vec<Vec<f64>>` so the
//! Gottlieb inner loop hits one cache-line per row instead of one
//! per `(n, m)` access. Bit-identity is preserved: the same `f64`
//! values are written in the same order — only the storage layout
//! differs.

use astrodyn_quantities::dims::GravParam;
use astrodyn_quantities::frame::SelfPlanet;
use uom::si::f64::Length;

/// Flat-storage triangular index: `(n, m) -> n*(n+1)/2 + m`.
///
/// Maps the `(degree, order)` pair to a flat `Vec<f64>` slot. The
/// formula equals the count of slots in rows `0..n` plus the offset
/// `m` within row `n`. Rows are therefore stored contiguously, which
/// keeps the Gottlieb inner loop on a single cache line per row.
#[inline]
pub(crate) const fn tri_idx(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m
}

/// Number of slots needed for a flat triangular array up to and
/// including row `degree`. Equal to `(degree + 1) * (degree + 2) / 2`.
#[inline]
const fn tri_len(degree: usize) -> usize {
    (degree + 1) * (degree + 2) / 2
}

/// Spherical harmonics gravity model data.
///
/// Holds the normalized gravity coefficients (Cnm, Snm) and precomputed
/// Gottlieb (1993) helper arrays for efficient gravity computation.
///
/// Ported from JEOD `SphericalHarmonicsGravitySource` + `initialize_body()`.
#[derive(Debug, Clone)]
pub struct SphericalHarmonicsData {
    /// Maximum degree of the model.
    pub degree: usize,
    /// Maximum order of the model.
    pub order: usize,
    /// Reference radius (m).
    pub radius: f64,
    /// Gravitational parameter (m^3/s^2).
    pub mu: f64,
    /// Normalized cosine coefficients, flat triangular storage
    /// indexed by `tri_idx(n, m)`. Access via [`Self::cnm`] /
    /// [`Self::cnm_row`].
    pub(crate) cnm: Vec<f64>,
    /// Normalized sine coefficients, flat triangular storage
    /// indexed by `tri_idx(n, m)`. Access via [`Self::snm`] /
    /// [`Self::snm_row`].
    pub(crate) snm: Vec<f64>,
    /// Whether C20 is tide-free.
    pub tide_free: bool,
    /// Delta to add to C20 to remove permanent tide.
    pub tide_free_delta: f64,

    // Precomputed Gottlieb helper arrays (from initialize_body()).
    // Flat triangular storage, indexed by `tri_idx(n, m)`.
    pub(crate) alpha: Vec<f64>,
    pub(crate) beta: Vec<f64>,
    pub(crate) xi: Vec<f64>,
    pub(crate) eta: Vec<f64>,
    pub(crate) zeta: Vec<f64>,
    pub(crate) upsilon: Vec<f64>,
    pub(crate) nrdiag: Vec<f64>,
    pub(crate) int_to_double: Vec<f64>,
}

impl SphericalHarmonicsData {
    /// Create and initialize a spherical harmonics model.
    ///
    /// Precomputes all Gottlieb helper arrays, matching JEOD's
    /// `SphericalHarmonicsGravitySource::initialize_body()`.
    ///
    /// `cnm` and `snm` accept the natural triangular `Vec<Vec<f64>>`
    /// shape (`cnm[n].len() == n + 1`) and are flattened internally
    /// into `(n, m) -> n*(n+1)/2 + m` storage.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        degree: usize,
        order: usize,
        radius: f64,
        mu: f64,
        cnm: Vec<Vec<f64>>,
        snm: Vec<Vec<f64>>,
        tide_free: bool,
        tide_free_delta: f64,
    ) -> Self {
        assert!(degree > 0, "degree must be > 0");
        assert!(order <= degree, "order must be <= degree");
        assert_eq!(cnm.len(), degree + 1);
        assert_eq!(snm.len(), degree + 1);
        for (n, row) in cnm.iter().enumerate() {
            assert_eq!(
                row.len(),
                n + 1,
                "cnm[{n}] must have {} elements, got {}",
                n + 1,
                row.len()
            );
        }
        for (n, row) in snm.iter().enumerate() {
            assert_eq!(
                row.len(),
                n + 1,
                "snm[{n}] must have {} elements, got {}",
                n + 1,
                row.len()
            );
        }

        let tri_n = tri_len(degree);
        let mut cnm_flat = vec![0.0_f64; tri_n];
        let mut snm_flat = vec![0.0_f64; tri_n];
        for n in 0..=degree {
            let base = tri_idx(n, 0);
            cnm_flat[base..base + n + 1].copy_from_slice(&cnm[n]);
            snm_flat[base..base + n + 1].copy_from_slice(&snm[n]);
        }

        let mut data = Self {
            degree,
            order,
            radius,
            mu,
            cnm: cnm_flat,
            snm: snm_flat,
            tide_free,
            tide_free_delta,
            alpha: vec![0.0; degree + 1],
            beta: vec![0.0; degree + 1],
            xi: vec![0.0; tri_n],
            eta: vec![0.0; tri_n],
            zeta: vec![0.0; tri_n],
            upsilon: vec![0.0; tri_n],
            nrdiag: vec![0.0; degree + 1],
            int_to_double: vec![0.0; degree + 2],
        };

        data.initialize_body();
        data
    }

    /// Precompute Gottlieb helper arrays.
    /// Direct port of JEOD `spherical_harmonics_gravity_source.cc::initialize_body()`.
    #[allow(clippy::needless_range_loop)]
    fn initialize_body(&mut self) {
        let degree = self.degree;

        // int_to_double[ii] = ii as f64
        for ii in 0..=degree + 1 {
            self.int_to_double[ii] = ii as f64;
        }

        // Pnm is only needed during initialization (to compute nrdiag).
        // Pnm[ii] has ii+3 elements.
        let mut pnm: Vec<Vec<f64>> = Vec::with_capacity(degree + 1);
        for ii in 0..=degree {
            pnm.push(vec![0.0; ii + 3]);
        }

        // Bottom of page 47 and page 48, and see equation (7-8)
        pnm[0][0] = 1.0;
        pnm[0][1] = 0.0;
        pnm[0][2] = 0.0;
        pnm[1][1] = 3.0_f64.sqrt();
        pnm[1][2] = 0.0;
        pnm[1][3] = 0.0;

        let i2d = &self.int_to_double;

        // Pages 46-47
        for ii in 2..=degree {
            let ii_f = i2d[ii];

            for jj in 0..=(ii - 1) {
                let jj_f = i2d[jj];
                // Equation (7-10)
                let num1 = (2.0 * ii_f - 1.0) * (2.0 * ii_f + 1.0);
                let den1 = (ii_f + jj_f) * (ii_f - jj_f);
                self.xi[tri_idx(ii, jj)] = (num1 / den1).sqrt();

                // Equation (7-10)
                let num2 = (2.0 * ii_f + 1.0) * (ii_f + jj_f - 1.0) * (ii_f - jj_f - 1.0);
                let den2 = (ii_f + jj_f) * (ii_f - jj_f) * (2.0 * ii_f - 3.0);
                if num2 == 0.0 {
                    self.eta[tri_idx(ii, jj)] = 0.0;
                } else {
                    self.eta[tri_idx(ii, jj)] = (num2 / den2).sqrt();
                }
            }

            for jj in 0..=ii {
                let jj_f = i2d[jj];
                if ii == jj {
                    self.zeta[tri_idx(ii, jj)] = 0.0;
                    self.upsilon[tri_idx(ii, jj)] = 0.0;
                } else if jj == 0 {
                    // Equation (7-19)
                    self.zeta[tri_idx(ii, 0)] = (ii_f * (ii_f + 1.0) / 2.0).sqrt();
                    // Equation (7-22)
                    self.upsilon[tri_idx(ii, 0)] =
                        (ii_f * (ii_f - 1.0) * (ii_f + 1.0) * (ii_f + 2.0) / 2.0).sqrt();
                } else {
                    // Equation (7-19)
                    self.zeta[tri_idx(ii, jj)] = ((ii_f - jj_f) * (ii_f + jj_f + 1.0)).sqrt();
                    // Equation (7-22)
                    self.upsilon[tri_idx(ii, jj)] = ((ii_f - jj_f)
                        * (ii_f + jj_f + 1.0)
                        * (ii_f - jj_f - 1.0)
                        * (ii_f + jj_f + 2.0))
                        .sqrt();
                }
            }

            // P(n,n) term, equation (7-8)
            pnm[ii][ii] = ((2.0 * ii_f + 1.0) / (2.0 * ii_f)).sqrt() * pnm[ii - 1][ii - 1];

            // P(n,n+1) and P(n,n+2) terms, table 1 (p. 14)
            pnm[ii][ii + 1] = 0.0;
            pnm[ii][ii + 2] = 0.0;

            // Equation (7-15) and (7-16)
            self.nrdiag[ii] = (2.0 * ii_f + 1.0).sqrt() * pnm[ii - 1][ii - 1];

            // Equation (7-13)
            self.alpha[ii] = ((2.0 * ii_f + 1.0) * (2.0 * ii_f - 1.0)).sqrt() / ii_f;
            self.beta[ii] = ((2.0 * ii_f + 1.0) / (2.0 * ii_f - 3.0)).sqrt() * (ii_f - 1.0) / ii_f;
        }
    }

    /// Read a Cnm coefficient: `cnm[n][m]` in the original
    /// `Vec<Vec<f64>>` layout, now backed by flat triangular storage
    /// indexed by `n*(n+1)/2 + m`. Panics if `m > n` (out of
    /// triangle): the triangular invariant must hold in release
    /// builds, since a stray `m > n` access would otherwise read a
    /// neighbouring `(n+1, …)` coefficient and silently return a
    /// physically wrong gravity-field value.
    #[inline]
    pub fn cnm(&self, n: usize, m: usize) -> f64 {
        assert!(
            m <= n,
            "cnm({n}, {m}): order m must be <= degree n (triangular index)"
        );
        self.cnm[tri_idx(n, m)]
    }

    /// Read an Snm coefficient: `snm[n][m]` in the original
    /// `Vec<Vec<f64>>` layout, now backed by flat triangular storage
    /// indexed by `n*(n+1)/2 + m`. Panics if `m > n` (out of
    /// triangle): see [`Self::cnm`] for why this is a release-build
    /// `assert!` rather than `debug_assert!`.
    #[inline]
    pub fn snm(&self, n: usize, m: usize) -> f64 {
        assert!(
            m <= n,
            "snm({n}, {m}): order m must be <= degree n (triangular index)"
        );
        self.snm[tri_idx(n, m)]
    }

    /// Borrow row `n` of the cosine coefficients as a contiguous
    /// `&[f64]` of length `n + 1`. Rows are stored contiguously in
    /// flat triangular storage, so this is a zero-copy reslice.
    #[inline]
    pub fn cnm_row(&self, n: usize) -> &[f64] {
        let base = tri_idx(n, 0);
        &self.cnm[base..base + n + 1]
    }

    /// Borrow row `n` of the sine coefficients as a contiguous
    /// `&[f64]` of length `n + 1`.
    #[inline]
    pub fn snm_row(&self, n: usize) -> &[f64] {
        let base = tri_idx(n, 0);
        &self.snm[base..base + n + 1]
    }

    /// Typed accessor for the gravitational parameter μ.
    ///
    /// Returns [`GravParam<SelfPlanet>`] (m³/s²) — the same numeric value as
    /// the public `mu` field, just with the dimensional annotation
    /// attached. The planet phantom is [`SelfPlanet`] because the
    /// source data is keyed by runtime ID (the central body identity is
    /// not load-bearing in this struct's static type). Mission code that
    /// knows the planet at compile time can relabel via
    /// [`GravParam::relabel`] at the call site.
    #[inline]
    pub fn mu_typed(&self) -> GravParam<SelfPlanet> {
        GravParam::<SelfPlanet>::from_si(self.mu)
    }

    /// Typed accessor for the reference radius.
    ///
    /// Returns [`Length`] (m).
    #[inline]
    pub fn radius_typed(&self) -> Length {
        Length::new::<uom::si::length::meter>(self.radius)
    }
}
