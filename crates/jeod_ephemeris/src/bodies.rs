/// Solar system body identifiers for ephemeris queries.
///
/// Matches JEOD's DE4xx item ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EphemerisBody {
    Mercury,
    Venus,
    EarthMoonBarycenter,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
    Moon,
    Sun,
    Earth,
}
