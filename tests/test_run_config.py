import hashlib
import json
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import yaml

import simsit.run_config as run_config
import simsit.cli as cli
from simsit.catalog import GaiaStarCatalog
from simsit.simulate import _RunLifecycle, make_sky_flat
from simsit.run_config import (
    _canonical_json,
    build_provenance,
    fingerprint_inputs,
    prepare_output,
    validate_config,
    write_provenance,
)
from simsit.simulate import simulate


@pytest.fixture
def config():
    return {
        "outdir": "output/example",
        "n_obs": 3,
        "n_example": 1,
        "n_sat": 1,
        "seed": 7,
        "instrument": {
            "image_shape": [32, 24],
            "pixel_scale": 1.0,
            "gain": 2.0,
            "read_noise": 3.0,
            "quantum_efficiency": 0.8,
            "aperture": 1.0,
            "obscuration": 0.1,
            "distortion": {"th": [0.0, 1.0], "dthdr": [1.0, 0.9]},
            "vignetting": {"th": [0.0, 1.0], "unvig": [1.0, 0.5]},
            "passband": {
                "name": "example",
                "wavelength_nm": [500.0, 600.0, 700.0],
                "throughput": [0.1, 0.8, 0.2],
                "reference": "AB",
            },
        },
        "conditions": {
            "psf_fwhm_range": [2.0, 4.0],
            "sky_range": [18.0, 20.0],
            "zp_range": [0.0, 0.3],
        },
        "cadence": {"exptime": [0.1, 0.2]},
        "tracker": {
            "type": "sidereal",
            "rot_sky_pos0": 0.0,
            "error": {"boresight": 0.2, "v_perturb": 1.0},
        },
        "sites": ["test"],
        "sat": {"mag_range": [10.0, 13.0]},
        "catalog": {"develop": True, "min_snr": 5.0},
        "photometry": {"model": "constant"},
    }


def test_validate_config_deepcopies(config):
    original = deepcopy(config)
    validated = validate_config(config)

    assert validated == config
    assert validated is not config
    validated["instrument"]["image_shape"][0] = 99
    assert config == original


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("n_obs", 0),
        ("n_example", 4),
        ("n_sat", 3),
        ("seed", -1),
        ("cadence.exptime", [0.0]),
        ("conditions.psf_fwhm_range", [4.0, 2.0]),
        ("instrument.gain", float("inf")),
        ("instrument.obscuration", 1.0),
        ("instrument.vignetting.unvig", [1.0, 1.2]),
        ("tracker.type", "bad"),
    ],
)
def test_validation_rejects_invalid_physical_values(config, path, value):
    candidate = deepcopy(config)
    target = candidate
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError, match=parts[-1]):
        validate_config(candidate)


def test_optional_models_are_validated(config):
    config["sat"]["mag_range"] = None
    config["photometry"] = {
        "model": "lambertian",
        "params": {"radius_m": 1.0, "albedo": 0.3},
        "spectral_relative_tolerance": 1e-4,
        "max_spectral_samples": 128,
    }
    config["integration"] = {
        "pixel_tolerance": 1e-3,
        "flux_relative_tolerance": 1e-4,
        "max_step_s": None,
        "max_depth": 0,
        "max_samples": 33,
    }
    config["observation"] = {"mode": "apparent"}
    assert validate_config(config)["instrument"]["passband"]["name"] == "example"


def test_integration_false_is_disabled(config):
    config["integration"] = False
    assert validate_config(config)["integration"] is False


@pytest.mark.parametrize(
    ("integration", "message"),
    [
        ({"max_depth": -1}, "max_depth"),
        ({"max_samples": 4}, "max_samples"),
        ({"max_step_s": 0.001, "max_samples": 100}, "max_samples"),
    ],
)
def test_integration_rejects_insufficient_sample_budget(config, integration, message):
    config["integration"] = integration
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_faceted_photometry_schema_is_explicit(config):
    config["sat"]["mag_range"] = None
    config["photometry"] = {
        "model": "faceted",
        "params": {
            "facets": [{
                "area": 2.0,
                "normal_body": [1.0, 0.0, 0.0],
                "center_of_pressure": [0.0, 0.0, 0.0],
                "diffuse_reflectivity": 0.5,
                "name": "panel",
            }],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        },
    }
    assert validate_config(config)["photometry"]["model"] == "faceted"


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("constant", {"radius_m": 1.0}),
        ("lambertian", {"radius_m": 1.0}),
        ("faceted", {"quaternion": [1.0, 0.0, 0.0, 0.0]}),
    ],
)
def test_photometry_rejects_model_specific_parameter_errors(config, model, params):
    config["sat"]["mag_range"] = None
    config["photometry"] = {"model": model, "params": params}
    with pytest.raises(ValueError, match="photometry.params"):
        validate_config(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("instrument.passband.wavelength_nm", [500.0]),
        ("instrument.passband.wavelength_nm", [500.0, 400.0]),
        ("instrument.passband.throughput", [0.0, 0.0, 0.0]),
        ("instrument.passband.throughput", [0.2, 0.3]),
        ("photometry.model", "unknown"),
        ("photometry.spectral_relative_tolerance", 0.0),
        ("photometry.max_spectral_samples", True),
        ("integration.flux_relative_tolerance", 1.1),
        ("observation.mode", "unknown"),
    ],
)
def test_canonical_optional_contract_rejects_invalid_values(config, path, value):
    candidate = deepcopy(config)
    parts = path.split(".")
    if parts[0] == "photometry":
        candidate["sat"]["mag_range"] = None
        candidate["photometry"] = {
            "model": "lambertian",
            "params": {"radius_m": 1.0, "albedo": 0.3},
        }
    if parts[0] == "observation":
        candidate["observation"] = {"mode": "apparent"}
    target = candidate
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    with pytest.raises(ValueError, match=parts[-1].split("_")[0]):
        validate_config(candidate)


def test_provenance_hashes_streamed_input_and_redacts_secret(tmp_path, config):
    input_file = tmp_path / "abc.bin"
    input_file.write_bytes(b"abc")
    config["meta"] = {"label": "synthetic", "count": 1}
    provenance = build_provenance(config, [input_file])

    assert provenance["schema_version"] == 1
    assert provenance["source"]["module"] == "simsit.run_config"
    assert len(provenance["source"]["source_sha256"]) == 64
    assert "run_config.py" in provenance["source"]["module_sha256"]
    assert provenance["input_sha256"][0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert provenance["input_sha256"][0]["path"] == "input.bin"
    canonical = _canonical_json(config).decode()
    assert "synthetic" not in canonical
    assert str(input_file) not in str(provenance)
    assert provenance["config_sha256"] == build_provenance(config)["config_sha256"]


@pytest.mark.parametrize("key", ["signature", "passwd", "cookie", "session", "jwt"])
def test_provenance_redacts_credential_key_variants(config, key):
    config[key] = "FAKE_SECRET_NOT_REAL"
    assert "FAKE_SECRET_NOT_REAL" not in _canonical_json(config).decode()


def test_metadata_rejects_credentials_and_personal_paths(config):
    for value in (
        {"password": "FAKE_SECRET_NOT_REAL"},
        {"notebook_path": "/home/private/notebook.ipynb"},
        {"nested": {"value": 1}},
    ):
        config["meta"] = value
        with pytest.raises(ValueError, match="meta"):
            validate_config(config)


def test_provenance_changes_when_a_source_module_changes(monkeypatch, config):
    original = run_config._file_sha256

    def changed(path):
        digest = original(path)
        if path.name == "tracker.py":
            return "0" * 64
        return digest

    baseline = build_provenance(config)
    monkeypatch.setattr(run_config, "_file_sha256", changed)
    changed_result = build_provenance(config)
    assert (
        baseline["source"]["module_sha256"]["tracker.py"]
        != changed_result["source"]["module_sha256"]["tracker.py"]
    )


@pytest.mark.parametrize("filename", [".env", "settings.pem", "id_rsa", "id_ed25519"])
def test_provenance_rejects_credential_input_files(tmp_path, config, filename):
    credential_file = tmp_path / filename
    credential_file.write_bytes(b"FAKE_PRIVATE_MATERIAL_NOT_REAL")
    with pytest.raises(ValueError, match="credential-like"):
        build_provenance(config, [credential_file])


def test_provenance_rejects_credential_loader_path(tmp_path, config):
    credential_dir = tmp_path / "credential-loader"
    credential_dir.mkdir()
    input_file = credential_dir / "catalog.dat"
    input_file.write_bytes(b"FAKE_PRIVATE_MATERIAL_NOT_REAL")
    with pytest.raises(ValueError, match="credential-like"):
        build_provenance(config, [input_file])


def test_provenance_rejects_url_userinfo(config):
    config["instrument"]["passband"]["reference"] = (
        "https://fake-user:FAKE_URL_SECRET@example.invalid/catalog"
    )
    with pytest.raises(ValueError, match="Calibration references"):
        build_provenance(config)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.invalid/catalog?citation=1",
        "https://example.invalid/catalog#section",
    ],
)
def test_provenance_rejects_url_credential_query_and_fragment(config, value):
    config["instrument"]["passband"]["reference"] = value
    with pytest.raises(ValueError, match="query/fragment"):
        build_provenance(config)


def test_provenance_resolves_and_rejects_credential_symlink(tmp_path, config):
    credential = tmp_path / ".env"
    credential.write_text("FAKE_PRIVATE_MATERIAL_NOT_REAL")
    link = tmp_path / "data.csv"
    link.symlink_to(credential)
    with pytest.raises(ValueError, match="credential-like"):
        build_provenance(config, [link])


def test_provenance_sorts_resolved_inputs(tmp_path, config):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.dat"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    result = build_provenance(config, [second, first])
    assert [item["path"] for item in result["input_sha256"]] == sorted(
        ["input.bin", "input.dat"]
    )


def test_actual_gaia_file_is_fingerprinted(tmp_path, monkeypatch):
    path = tmp_path / "1.fits"
    path.write_bytes(b"gaia-fits-input")
    data = Table({
        "coord_ra": [0.0] * u.rad,
        "coord_dec": [0.0] * u.rad,
        "pm_ra": [0.0] * u.mas / u.yr,
        "pm_dec": [0.0] * u.mas / u.yr,
        "parallax": [0.0] * u.arcsec,
        "phot_g_mean_mag": [15.0],
    })
    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: data)
    monkeypatch.setattr(
        "simsit.catalog.backend.catalog_to_apparent",
        lambda ra, dec, time, **kwargs: (np.asarray(ra), np.asarray(dec)),
    )
    catalog = GaiaStarCatalog(tmp_path)
    catalog._get_trixel_stars(1, Time("J2000"))
    result = fingerprint_inputs(catalog.input_paths)
    assert result[0]["sha256"] == hashlib.sha256(
        b"gaia-fits-input"
    ).hexdigest()


def test_provenance_does_not_persist_output_or_metadata_paths(config):
    config["outdir"] = "/home/private/notebook/output"
    config["image_dir"] = "images"
    config["meta"] = {"label": "safe"}
    result = build_provenance(config)
    assert "/home/private/notebook/output" not in str(result)
    assert "safe" not in str(result)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("instrument", "gain_typo"),
        ("conditions", "sky_rnage"),
        ("tracker", "rot_sky_pos_typo"),
        ("catalog", "min_snr_typo"),
        ("observation", "mode_typo"),
        ("integration", "max_depth_typo"),
        ("photometry", "model_typo"),
    ],
)
def test_validation_rejects_unknown_nested_keys(config, section, key):
    config[section] = deepcopy(config.get(section, {}))
    config[section][key] = 1
    if section == "photometry":
        config[section]["model"] = "constant"
    with pytest.raises(ValueError, match="unknown key"):
        validate_config(config)


def test_validation_rejects_unknown_top_level_keys(config):
    config["outdir_typo"] = "private/notebook"
    with pytest.raises(ValueError, match="unknown key"):
        validate_config(config)


@pytest.mark.parametrize("key", ["image_dir", "reference_dir"])
@pytest.mark.parametrize("value", ["/tmp/images", "../images", "images/../other", "C:\\images"])
def test_validation_requires_contained_relative_output_directories(config, key, value):
    config[key] = value
    with pytest.raises(ValueError, match="relative"):
        validate_config(config)


def test_validation_requires_current_passband_and_catalog_contract(config):
    passband = deepcopy(config["instrument"]["passband"])
    config["instrument"].pop("passband")
    with pytest.raises(ValueError, match="passband"):
        validate_config(config)

    config = deepcopy(config)
    config["instrument"]["passband"] = passband
    config["catalog"]["develop"] = False
    config["catalog"]["gaia_dir"] = "catalog"
    with pytest.raises(ValueError, match="magnitude_column"):
        validate_config(config)


def test_validation_matches_calibration_and_empirical_zeropoint_contract(config):
    config["instrument"]["passband"]["reference"] = (
        "https://example.invalid/curve?citation=1"
    )
    assert validate_config(config)["instrument"]["passband"]["reference"].endswith(
        "citation=1"
    )

    config["instrument"]["zeropoint"] = 20.0
    config["instrument"].pop("zeropoint_reference", None)
    with pytest.raises(ValueError, match="zeropoint_reference"):
        validate_config(config)

    config["instrument"]["zeropoint_reference"] = (
        "https://example.invalid/calibration"
    )
    assert validate_config(config)["instrument"]["zeropoint"] == 20.0


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_n_sat_is_strict_integer(config, value):
    config["n_sat"] = value
    with pytest.raises(ValueError, match="n_sat"):
        validate_config(config)


@pytest.mark.parametrize(
    ("tracker_type", "missing"),
    [("orbit", "t_rewind"), ("orbit", "v_perturb"), ("sidereal", "boresight")],
)
def test_tracker_requires_type_specific_errors(config, tracker_type, missing):
    config["tracker"]["type"] = tracker_type
    if tracker_type == "orbit":
        config["tracker"]["error"].update({"t_rewind": 0.0, "v_perturb": 0.0})
    config["tracker"]["error"].pop(missing, None)
    with pytest.raises(ValueError, match=missing):
        validate_config(config)


def test_prepare_output_is_non_destructive(tmp_path):
    missing = tmp_path / "new" / "run"
    assert prepare_output(missing) == missing
    assert missing.is_dir() and not list(missing.iterdir())

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare_output(existing)
    assert sentinel.read_text() == "preserve"

    assert prepare_output(existing, overwrite=True) == existing
    assert sentinel.read_text() == "preserve"


def test_prepare_output_rejects_symlink_escape(tmp_path):
    root = tmp_path / "output"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "images").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_output(root, overwrite=True, relative_dirs=("images",))


def test_simulate_rejects_all_existing_output_symlinks(tmp_path, monkeypatch, config):
    root = tmp_path / "run"
    images = root / "images"
    reference = root / "reference"
    images.mkdir(parents=True)
    reference.mkdir()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("do not overwrite")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    dangling_target = tmp_path / "missing-target"
    (images / "sample_metadata.yaml").symlink_to(sentinel)
    (images / "0001.fits").symlink_to(sentinel)
    (reference / "dangling.fits").symlink_to(dangling_target)
    (reference / "unrelated-directory").symlink_to(
        outside_directory, target_is_directory=True
    )
    old_manifest = root / "provenance.json"
    old_manifest.write_text('{"status":"complete","products":[]}\n')
    config["outdir"] = str(root)
    site = Observer(
        location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg),
        name="test",
    )
    monkeypatch.setattr(Observer, "at_site", lambda name: site)
    with pytest.raises(ValueError, match="symlink"):
        simulate(config, overwrite=True)
    assert sentinel.read_text() == "do not overwrite"
    assert old_manifest.read_text() == '{"status":"complete","products":[]}\n'
    assert not (images / "reference_truth_1.fits").exists()


def test_prepare_output_preflights_all_relative_dirs(tmp_path):
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="relative"):
        prepare_output(root, relative_dirs=("images", "../escape"))
    assert not root.exists()


def test_write_provenance_is_atomic_json(tmp_path):
    target = write_provenance(tmp_path, {"source": {"sha256": "abc"}})
    assert target == tmp_path / "provenance.json"
    assert target.read_text() == '{"source":{"sha256":"abc"}}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_lifecycle_product_hashing_is_linear(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    calls = []
    original = __import__("simsit.simulate", fromlist=["_file_sha256"])._file_sha256

    def counted(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr("simsit.simulate._file_sha256", counted)
    lifecycle = _RunLifecycle(root, {}, (), False)
    lifecycle.start()
    products = []
    for index in range(8):
        product = root / f"product-{index}.dat"
        product.write_bytes(str(index).encode())
        products.append(product)
        lifecycle.record_product(product)
    lifecycle.complete()
    assert len(calls) == len(products)


def test_lifecycle_reuses_resolved_input_paths(tmp_path, monkeypatch, config):
    input_file = tmp_path / "catalog.fits"
    input_file.write_bytes(b"input")
    root = tmp_path / "run"
    root.mkdir()
    original_resolve = Path.resolve
    calls = []

    def counted(path, *args, **kwargs):
        calls.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted)
    lifecycle = _RunLifecycle(root, {}, (input_file,), False)
    initial_calls = len(calls)
    lifecycle.record_inputs((input_file,))
    lifecycle.record_inputs((input_file,))
    assert len(calls) == initial_calls


def _sky_flat_config(tmp_path):
    root = tmp_path / "run"
    images = root / "images"
    images.mkdir(parents=True)
    fits.PrimaryHDU(np.ones((2, 2))).writeto(images / "0001.fits")
    config = {
        "outdir": str(root),
        "image_dir": "images",
        "seed": 1,
        "instrument": {"image_shape": [2, 2]},
    }
    configfile = tmp_path / "sky-flat.yaml"
    configfile.write_text(yaml.safe_dump(config))
    return configfile, root, images


def test_make_sky_flat_requires_explicit_overwrite_and_preserves_unrelated(
    tmp_path,
):
    configfile, root, images = _sky_flat_config(tmp_path)
    unrelated = root / "unrelated.txt"
    unrelated.write_text("preserve")
    target = images / "sky_flat.fits"
    target.write_bytes(b"old-flat")

    with pytest.raises(FileExistsError):
        make_sky_flat(configfile)
    assert target.read_bytes() == b"old-flat"
    assert unrelated.read_text() == "preserve"

    make_sky_flat(configfile, overwrite=True)
    assert target.read_bytes() != b"old-flat"
    assert unrelated.read_text() == "preserve"


def test_make_sky_flat_rejects_root_and_product_symlinks(tmp_path):
    configfile, root, images = _sky_flat_config(tmp_path)
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("preserve")
    product = images / "sky_flat.fits"
    product.symlink_to(sentinel)
    with pytest.raises(ValueError, match="symlink"):
        make_sky_flat(configfile, overwrite=True)
    assert sentinel.read_text() == "preserve"
    assert product.is_symlink()

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    link_root = tmp_path / "link-root"
    link_root.symlink_to(real_root, target_is_directory=True)
    configfile.write_text(
        yaml.safe_dump({
            "outdir": str(link_root),
            "image_dir": "images",
            "seed": 1,
            "instrument": {"image_shape": [2, 2]},
        })
    )
    with pytest.raises(ValueError, match="symlink"):
        make_sky_flat(configfile, overwrite=True)
    assert sentinel.read_text() == "preserve"


def test_simulate_preflights_and_writes_manifest(tmp_path, monkeypatch, config):
    config["outdir"] = str(tmp_path / "run")
    config["photometry"] = {"model": "constant"}
    config["sat"]["mag_range"] = None
    site = Observer(
        location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg),
        name="test",
    )
    monkeypatch.setattr(Observer, "at_site", lambda name: site)
    monkeypatch.setattr("simsit.simulate.tqdm", lambda iterable: [])

    simulate(config)

    manifest = tmp_path / "run" / "provenance.json"
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "complete"
    assert payload["products"]
    assert payload["stale_outputs_retained"] is False
    assert (tmp_path / "run" / "images").is_dir()
    assert (tmp_path / "run" / "reference").is_dir()


def test_simulate_rejects_invalid_config_before_output(tmp_path, config):
    config["outdir"] = str(tmp_path / "run")
    config["image_dir"] = "../escape"
    with pytest.raises(ValueError, match="relative"):
        simulate(config)
    assert not (tmp_path / "run").exists()


def test_simulate_rejects_invalid_integration_before_output(tmp_path, config):
    config["outdir"] = str(tmp_path / "run")
    config["integration"] = {"max_step_s": 0.01, "max_samples": 10}
    with pytest.raises(ValueError, match="max_samples"):
        simulate(config)
    assert not (tmp_path / "run").exists()


def test_simulate_overwrite_preserves_unrelated_files(tmp_path, monkeypatch, config):
    config["outdir"] = str(tmp_path / "run")
    config["photometry"] = {"model": "constant"}
    config["sat"]["mag_range"] = None
    site = Observer(
        location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg),
        name="test",
    )
    monkeypatch.setattr(Observer, "at_site", lambda name: site)
    monkeypatch.setattr("simsit.simulate.tqdm", lambda iterable: [])
    simulate(config)
    sentinel = tmp_path / "run" / "unrelated.txt"
    sentinel.write_text("preserve")

    with pytest.raises(FileExistsError):
        simulate(config)
    assert sentinel.read_text() == "preserve"

    simulate(config, overwrite=True)
    assert sentinel.read_text() == "preserve"


def test_simulate_failed_run_writes_failed_manifest(tmp_path, monkeypatch, config):
    config["outdir"] = str(tmp_path / "run")
    site = Observer(
        location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg),
        name="test",
    )
    monkeypatch.setattr(Observer, "at_site", lambda name: site)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic lifecycle failure")

    monkeypatch.setattr("simsit.simulate.tqdm", lambda iterable: [0])
    monkeypatch.setattr("simsit.simulate.make_image", fail)
    with pytest.raises(RuntimeError, match="synthetic lifecycle failure"):
        simulate(config)
    payload = json.loads(
        (tmp_path / "run" / "provenance.json").read_text()
    )
    assert payload["status"] == "failed"
    assert payload["products"] == []


def test_simulate_unknown_site_rejects_before_output(tmp_path, config):
    config["outdir"] = str(tmp_path / "run")
    config["sites"] = ["site-that-does-not-exist"]
    with pytest.raises(Exception, match="site-that-does-not-exist"):
        simulate(config)
    assert not (tmp_path / "run").exists()


def test_simulate_missing_gaia_directory_rejects_before_output(
    tmp_path, monkeypatch, config
):
    config["outdir"] = str(tmp_path / "run")
    config["catalog"].update({
        "develop": False,
        "gaia_dir": str(tmp_path / "missing-gaia"),
        "magnitude_column": "mag",
        "magnitude_system": "AB",
    })
    site = Observer(
        location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg),
        name="test",
    )
    monkeypatch.setattr(Observer, "at_site", lambda name: site)
    with pytest.raises(FileNotFoundError):
        simulate(config)
    assert not (tmp_path / "run").exists()


def test_manifest_excludes_stale_products(tmp_path, monkeypatch, config):
    config["outdir"] = str(tmp_path / "run")
    config["photometry"] = {"model": "constant"}
    config["sat"]["mag_range"] = None
    site = Observer(
        location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg),
        name="test",
    )
    monkeypatch.setattr(Observer, "at_site", lambda name: site)
    monkeypatch.setattr("simsit.simulate.tqdm", lambda iterable: [])
    simulate(config)
    stale = tmp_path / "run" / "images" / "9999.fits"
    stale.write_bytes(b"stale")

    simulate(config, overwrite=True)
    payload = json.loads(
        (tmp_path / "run" / "provenance.json").read_text()
    )
    assert "images/9999.fits" not in {
        product["path"] for product in payload["products"]
    }
    assert payload["stale_outputs_retained"] is True
    assert "not current-run products" in payload["stale_outputs_note"]


def test_cli_passes_explicit_overwrite_and_config_input(tmp_path, monkeypatch):
    config_file = tmp_path / "run.yaml"
    config_file.write_text("n_obs: 1\n")
    calls = []

    def fake_simulate(config, *, overwrite, input_paths):
        calls.append((config, overwrite, tuple(input_paths)))

    monkeypatch.setattr(cli, "simulate", fake_simulate)
    assert cli.main([str(config_file), "--overwrite"]) == 0
    assert calls == [({"n_obs": 1}, True, (config_file,))]
