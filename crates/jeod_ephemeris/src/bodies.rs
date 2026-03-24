/// Solar system body identifiers for ephemeris queries.
///
/// Follows the JEOD/JPL DE4xx convention where planet names (items 0-8)
/// denote **system barycenters**, not the planet body centers. For example,
/// `Jupiter` is the Jupiter system barycenter (NAIF ID 5), not Jupiter's
/// body center (NAIF ID 599). The distinction matters for bodies with
/// large moons (Jupiter, Saturn, etc.) but is negligible for Mercury/Venus.
///
/// `Earth` and `Moon` are separate entries because JPL DE files resolve
/// them individually from the Earth-Moon barycenter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EphemerisBody {
    /// Solar system barycenter (NAIF ID 0). Origin of the ICRF.
    SolarSystemBarycenter,
    /// Mercury system barycenter (NAIF ID 1). Coincides with body center
    /// because Mercury has no moons.
    Mercury,
    /// Venus system barycenter (NAIF ID 2). Coincides with body center
    /// because Venus has no moons.
    Venus,
    /// Earth-Moon barycenter (NAIF ID 3). Located inside Earth, ~4670 km
    /// from Earth's center toward the Moon.
    EarthMoonBarycenter,
    /// Mars system barycenter (NAIF ID 4). Effectively coincides with
    /// Mars body center (Phobos and Deimos are negligible mass).
    Mars,
    /// Jupiter system barycenter (NAIF ID 5). Offset from Jupiter body
    /// center due to Galilean moons.
    Jupiter,
    /// Saturn system barycenter (NAIF ID 6). Offset from Saturn body
    /// center due to Titan and other moons.
    Saturn,
    /// Uranus system barycenter (NAIF ID 7).
    Uranus,
    /// Neptune system barycenter (NAIF ID 8).
    Neptune,
    /// Pluto system barycenter (NAIF ID 9). Significant offset from
    /// Pluto body center due to Charon.
    Pluto,
    /// Moon body center (NAIF ID 301).
    Moon,
    /// Sun body center (NAIF ID 10).
    Sun,
    /// Earth body center (NAIF ID 399).
    Earth,
}
