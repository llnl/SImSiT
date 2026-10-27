import galsim
import numpy as np
import pytest

from simsit import Instrument


def instrument(**kwargs):
    config = dict(
        image_shape=(400, 400), gain=1.0, read_noise=0.0,
        pixel_scale=1.0, aperture=1.0,
        distortion={"th": [0, 0.1, 0.2, 0.3], "dthdr": [1, 1, 1, 1]},
    )
    config.update(kwargs)
    return Instrument(**config)


@pytest.mark.parametrize("gain,qe,read_noise", [(1, 1, 0), (4, 0.65, 3)])
def test_noise_against_poisson_and_read_noise(gain, qe, read_noise):
    detector = instrument(gain=gain, quantum_efficiency=qe, read_noise=read_noise)
    image = galsim.Image(400, 400, init_value=100.0)
    detector.apply_noise(image, sky_phot=1000, rng=np.random.default_rng(123))
    expected_mean = 100 * qe / gain
    expected_variance = (100 * qe + read_noise**2) / gain**2
    np.testing.assert_allclose(image.array.mean(), expected_mean, rtol=0.003)
    np.testing.assert_allclose(image.array.var(), expected_variance, rtol=0.02)


def test_snr_in_electrons_is_independent_of_gain():
    for gain in (1, 6.2):
        detector = instrument(gain=gain, quantum_efficiency=0.5, read_noise=3)
        neff = 2.266 * 2**2
        area = neff + np.sqrt(neff / np.pi) * 10
        expected = 100 / np.sqrt(100 + (20 * 0.3 * 0.5 + 9) * area)
        actual = detector.streak_snr(
            nphot=200, length=10, psf_fwhm=2, sky_phot=20, exptime=0.3,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_two_sample_throughput_and_odd_detector_shape():
    detector = instrument(
        image_shape=(41, 39),
        vignetting={"th": [0, 0.3], "unvig": [1, 0.5]},
    )
    image = detector.init_image(sky_phot=10, exptime=0.1)
    assert image.array.shape == (39, 41)
    assert image.bounds.center == galsim.PositionI(0, 0)
    before = image.array.copy()
    detector.apply_vignetting(image)
    assert np.all(image.array > 0)
    assert np.all(image.array <= before)
    np.testing.assert_allclose(image(0, 0), before[19, 20], rtol=1e-7)
