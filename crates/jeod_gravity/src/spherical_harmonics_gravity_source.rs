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
    /// Normalized cosine coefficients: cnm[n][m] for n=0..degree, m=0..n.
    pub cnm: Vec<Vec<f64>>,
    /// Normalized sine coefficients: snm[n][m] for n=0..degree, m=0..n.
    pub snm: Vec<Vec<f64>>,
    /// Whether C20 is tide-free.
    pub tide_free: bool,
    /// Delta to add to C20 to remove permanent tide.
    pub tide_free_delta: f64,

    // Precomputed Gottlieb helper arrays (from initialize_body())
    pub(crate) alpha: Vec<f64>,
    pub(crate) beta: Vec<f64>,
    pub(crate) xi: Vec<Vec<f64>>,
    pub(crate) eta: Vec<Vec<f64>>,
    pub(crate) zeta: Vec<Vec<f64>>,
    pub(crate) upsilon: Vec<Vec<f64>>,
    pub(crate) nrdiag: Vec<f64>,
    pub(crate) int_to_double: Vec<f64>,
}

impl SphericalHarmonicsData {
    /// Create and initialize a spherical harmonics model.
    ///
    /// Precomputes all Gottlieb helper arrays, matching JEOD's
    /// `SphericalHarmonicsGravitySource::initialize_body()`.
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

        let mut data = Self {
            degree,
            order,
            radius,
            mu,
            cnm,
            snm,
            tide_free,
            tide_free_delta,
            alpha: vec![0.0; degree + 1],
            beta: vec![0.0; degree + 1],
            xi: Vec::with_capacity(degree + 1),
            eta: Vec::with_capacity(degree + 1),
            zeta: Vec::with_capacity(degree + 1),
            upsilon: Vec::with_capacity(degree + 1),
            nrdiag: vec![0.0; degree + 1],
            int_to_double: vec![0.0; degree + 2],
        };

        // Initialize xi, eta, zeta, upsilon with correct sizes
        for ii in 0..=degree {
            data.xi.push(vec![0.0; ii + 1]);
            data.eta.push(vec![0.0; ii + 1]);
            data.zeta.push(vec![0.0; ii + 1]);
            data.upsilon.push(vec![0.0; ii + 1]);
        }

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
                self.xi[ii][jj] = (num1 / den1).sqrt();

                // Equation (7-10)
                let num2 = (2.0 * ii_f + 1.0) * (ii_f + jj_f - 1.0) * (ii_f - jj_f - 1.0);
                let den2 = (ii_f + jj_f) * (ii_f - jj_f) * (2.0 * ii_f - 3.0);
                if num2 == 0.0 {
                    self.eta[ii][jj] = 0.0;
                } else {
                    self.eta[ii][jj] = (num2 / den2).sqrt();
                }
            }

            for jj in 0..=ii {
                let jj_f = i2d[jj];
                if ii == jj {
                    self.zeta[ii][jj] = 0.0;
                    self.upsilon[ii][jj] = 0.0;
                } else if jj == 0 {
                    // Equation (7-19)
                    self.zeta[ii][0] = (ii_f * (ii_f + 1.0) / 2.0).sqrt();
                    // Equation (7-22)
                    self.upsilon[ii][0] =
                        (ii_f * (ii_f - 1.0) * (ii_f + 1.0) * (ii_f + 2.0) / 2.0).sqrt();
                } else {
                    // Equation (7-19)
                    self.zeta[ii][jj] = ((ii_f - jj_f) * (ii_f + jj_f + 1.0)).sqrt();
                    // Equation (7-22)
                    self.upsilon[ii][jj] = ((ii_f - jj_f)
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
}
