import galsim
import numpy as np
import simsit


def test_mock_star_catalog():
    catalog = simsit.MockStarCatalog()
    rng = np.random.default_rng(57721)

    for _ in range(10):
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        ph = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(ph)
        y = r * np.sin(ph)

        center = galsim.CelestialCoord.from_xyz(x, y, z)
        time = None

        stars1 = catalog.get_stars(center, center, 0.1 * galsim.degrees, time)

        center2 = center.greatCirclePoint(
            galsim.CelestialCoord.from_xyz(0, 0, 1),
            0.09 * galsim.degrees,
        )
        stars2 = catalog.get_stars(center2, center2, 0.3 * galsim.degrees, time)

        # Every star in the smaller query should appear in the larger query.
        stars2_coords = set(zip(np.asarray(stars2["ra_rad"]), np.asarray(stars2["dec_rad"])))
        stars1_coords = set(zip(np.asarray(stars1["ra_rad"]), np.asarray(stars1["dec_rad"])))

        assert stars1_coords.issubset(stars2_coords)