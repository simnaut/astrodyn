//! 128-bit rational arithmetic for coefficient computation.
//!
//! Port of JEOD's `Ratio128` (er7_utils/math/include/ratio128.hh).
//! Uses Rust native `u128` instead of JEOD's `UInt128` wrapper.
//! Maintains exact rational arithmetic for coefficient generation,
//! then converts to f64 for runtime use.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// 128-bit rational number with exact arithmetic.
///
/// JEOD: `Ratio128` in `er7_utils/math/include/ratio128.hh`.
/// Fields: `sign` (-1, 0, 1), `num` (magnitude), `den` (magnitude).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ratio128 {
    sign: i8,
    num: u128,
    den: u128,
}

/// Compute GCD of two unsigned 128-bit integers (Euclidean algorithm).
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

impl Ratio128 {
    /// Zero.
    pub const ZERO: Self = Self {
        sign: 0,
        num: 0,
        den: 1,
    };

    /// One.
    pub const ONE: Self = Self {
        sign: 1,
        num: 1,
        den: 1,
    };

    /// Construct from a signed 64-bit integer.
    pub fn from_i64(val: i64) -> Self {
        if val == 0 {
            Self::ZERO
        } else if val > 0 {
            Self {
                sign: 1,
                num: val as u128,
                den: 1,
            }
        } else {
            Self {
                sign: -1,
                num: (-(val as i128)) as u128,
                den: 1,
            }
        }
    }

    /// Construct from unsigned numerator and denominator with sign.
    /// JEOD: `Ratio128(unsigned long long num_in, unsigned long long den_in, int sign_in)`.
    /// Faithful port of the JEOD constructor; currently exercised only by
    /// the unit tests in this module.
    #[allow(dead_code)]
    pub fn new(num: u128, den: u128, sign: i8) -> Self {
        assert!(den != 0, "Ratio128: zero denominator");
        if num == 0 {
            Self::ZERO
        } else {
            let mut r = Self { sign, num, den };
            r.reduce();
            r
        }
    }

    /// Reduce to canonical form via GCD.
    /// JEOD: `Ratio128::reduce()`.
    fn reduce(&mut self) {
        if self.num == 0 {
            self.sign = 0;
            self.den = 1;
            return;
        }
        let g = gcd(self.num, self.den);
        if g > 1 {
            self.num /= g;
            self.den /= g;
        }
    }

    /// Convert to f64.
    /// JEOD: `Ratio128::operator double() const` / `to_double()`.
    pub fn to_f64(self) -> f64 {
        if self.sign == 0 {
            return 0.0;
        }
        let value = (self.num as f64) / (self.den as f64);
        if self.sign < 0 {
            -value
        } else {
            value
        }
    }

    /// Multiplicative inverse.
    /// JEOD: `Ratio128::inverse()`.
    pub fn inverse(self) -> Self {
        assert!(self.num != 0, "Ratio128: inverse of zero");
        Self {
            sign: self.sign,
            num: self.den,
            den: self.num,
        }
    }
}

// ── Arithmetic operators ──
// All match JEOD's operator implementations with cross-GCD reduction.

impl Neg for Ratio128 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            sign: -self.sign,
            num: self.num,
            den: self.den,
        }
    }
}

impl Add for Ratio128 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if self.sign == 0 {
            return rhs;
        }
        if rhs.sign == 0 {
            return self;
        }
        // a/b + c/d = (a*d ± c*b) / (b*d)
        // Cross-reduce first to avoid overflow.
        let g1 = gcd(self.den, rhs.den);
        let left_den = self.den / g1;
        let right_den = rhs.den / g1;
        let common_den = left_den * rhs.den; // = self.den * right_den

        let left_num = self.num * right_den;
        let right_num = rhs.num * left_den;

        let (result_sign, result_num) = if self.sign == rhs.sign {
            (self.sign, left_num + right_num)
        } else {
            // Different signs: subtract magnitudes
            if left_num >= right_num {
                (self.sign, left_num - right_num)
            } else {
                (rhs.sign, right_num - left_num)
            }
        };

        if result_num == 0 {
            Self::ZERO
        } else {
            let mut r = Self {
                sign: result_sign,
                num: result_num,
                den: common_den,
            };
            r.reduce();
            r
        }
    }
}

impl Sub for Ratio128 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Mul for Ratio128 {
    type Output = Self;
    /// JEOD: `operator*=` with cross-GCD reduction before multiply.
    fn mul(self, rhs: Self) -> Self {
        if self.sign == 0 || rhs.sign == 0 {
            return Self::ZERO;
        }
        // Cross-reduce to prevent overflow:
        // (a/b) * (c/d) = (a/gcd(a,d)) * (c/gcd(c,b)) / ((b/gcd(c,b)) * (d/gcd(a,d)))
        let g_ad = gcd(self.num, rhs.den);
        let g_cb = gcd(rhs.num, self.den);
        let mut r = Self {
            sign: self.sign * rhs.sign,
            num: (self.num / g_ad) * (rhs.num / g_cb),
            den: (self.den / g_cb) * (rhs.den / g_ad),
        };
        r.reduce();
        r
    }
}

impl Div for Ratio128 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl AddAssign for Ratio128 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Ratio128 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Ratio128 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// ── Conversions ──

impl From<i32> for Ratio128 {
    fn from(val: i32) -> Self {
        Self::from_i64(val as i64)
    }
}

impl From<i64> for Ratio128 {
    fn from(val: i64) -> Self {
        Self::from_i64(val)
    }
}

impl From<u32> for Ratio128 {
    fn from(val: u32) -> Self {
        if val == 0 {
            Self::ZERO
        } else {
            Self {
                sign: 1,
                num: val as u128,
                den: 1,
            }
        }
    }
}

impl From<usize> for Ratio128 {
    fn from(val: usize) -> Self {
        if val == 0 {
            Self::ZERO
        } else {
            Self {
                sign: 1,
                num: val as u128,
                den: 1,
            }
        }
    }
}

/// Implement Ratio128 / integer for the recurrence relation.
impl Div<Ratio128> for i32 {
    type Output = Ratio128;
    fn div(self, rhs: Ratio128) -> Ratio128 {
        Ratio128::from(self) / rhs
    }
}

/// Copy of the `std::copy` pattern: convert Vec<Ratio128> → slice of f64.
impl Ratio128 {
    /// Convert a slice of Ratio128 to f64 values.
    pub fn slice_to_f64(src: &[Self], dst: &mut [f64]) {
        assert!(dst.len() >= src.len());
        for (i, r) in src.iter().enumerate() {
            dst[i] = r.to_f64();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Ratio128::from(3);
        let b = Ratio128::from(4);
        assert_eq!((a + b).to_f64(), 7.0);
        assert_eq!((a - b).to_f64(), -1.0);
        assert_eq!((a * b).to_f64(), 12.0);
        assert_eq!((a / b).to_f64(), 0.75);
    }

    #[test]
    fn test_rational_precision() {
        // 1/3 + 1/6 = 1/2
        let a = Ratio128::new(1, 3, 1);
        let b = Ratio128::new(1, 6, 1);
        let c = a + b;
        assert_eq!(c.to_f64(), 0.5);
        assert_eq!(c.num, 1);
        assert_eq!(c.den, 2);
    }

    #[test]
    fn test_zero_handling() {
        let z = Ratio128::ZERO;
        let a = Ratio128::from(5);
        assert_eq!((z + a).to_f64(), 5.0);
        assert_eq!((a + z).to_f64(), 5.0);
        assert_eq!((z * a).to_f64(), 0.0);
    }

    #[test]
    fn test_negation() {
        let a = Ratio128::from(3);
        assert_eq!((-a).to_f64(), -3.0);
        assert_eq!((a + (-a)).to_f64(), 0.0);
    }

    #[test]
    fn test_cross_reduction() {
        // (6/5) * (10/3) = 4, with cross-reduction preventing overflow
        let a = Ratio128::new(6, 5, 1);
        let b = Ratio128::new(10, 3, 1);
        let c = a * b;
        assert_eq!(c.to_f64(), 4.0);
    }

    #[test]
    fn test_adams_recurrence() {
        // c_0 = 1
        // c_1 = -c_0 / 2 = -1/2
        // c_2 = -(c_0/3 + c_1/2) = -(1/3 - 1/4) = -1/12
        let c0 = Ratio128::ONE;
        let c1 = -(c0 / Ratio128::from(2));
        assert_eq!(c1.to_f64(), -0.5);

        let c2 = -(c0 / Ratio128::from(3) + c1 / Ratio128::from(2));
        // c2 = -(1/3 - 1/4) = -(4/12 - 3/12) = -1/12
        assert!((c2.to_f64() - (-1.0 / 12.0)).abs() < 1e-15);
    }
}
