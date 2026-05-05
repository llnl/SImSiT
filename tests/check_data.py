import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


def image_id_from_filename(filename):
    """Extract the 4-digit image id from a FITS filename."""
    return os.path.splitext(os.path.basename(filename))[0]


def check_endpoints(path, file_pattern="????.fits", num_img=3, verbose=False):
    """
    Compare reference endpoint coordinates against image and WCS products.

    Parameters
    ----------
    path : str
        Output directory containing ``images`` and ``reference`` subdirectories.
    file_pattern : str, optional
        Glob pattern for image files.
    num_img : int, optional
        Number of random images to inspect.
    verbose : bool, optional
        If True, print additional FITS header information.
    """
    image_dir = os.path.join(path, "images")
    reference_dir = os.path.join(path, "reference")

    all_files = sorted(glob.glob(os.path.join(image_dir, file_pattern)))
    if len(all_files) == 0:
        raise ValueError(f"No image files found in {image_dir} matching {file_pattern!r}")

    n_select = min(num_img, len(all_files))
    files = np.random.choice(all_files, n_select, replace=False)

    os.makedirs("tests", exist_ok=True)
    pdf = PdfPages(os.path.join("tests", "check_endpoints.pdf"))

    reference_truth = fits.open(os.path.join(reference_dir, "reference_truth.fits"))

    for filename in files:
        print(f"*========== {filename} ==========*")

        image_id = image_id_from_filename(filename)

        img_hdul = fits.open(filename)
        img = img_hdul[0].data

        if verbose:
            print(filename)
            print(img_hdul[0].header)

        truth_wcs_hdul = fits.open(os.path.join(reference_dir, f"{image_id}.wcs.fits"))
        truth_wcst_hdul = fits.open(os.path.join(reference_dir, f"{image_id}.wcst.fits"))

        w_start = WCS(truth_wcs_hdul[0].header)
        w_end = WCS(truth_wcst_hdul[0].header)

        sat_hdu_name = f"SAT_{image_id}"
        sat_data = reference_truth[sat_hdu_name].data

        ra0 = sat_data.ra0[0]
        dec0 = sat_data.dec0[0]
        ra1 = sat_data.ra1[0]
        dec1 = sat_data.dec1[0]

        x0 = sat_data.x0_FITS[0]
        y0 = sat_data.y0_FITS[0]
        x1 = sat_data.x1_FITS[0]
        y1 = sat_data.y1_FITS[0]

        ra_dec_0_pix = w_start.wcs_world2pix([[ra0, dec0]], 1)
        ra_dec_1_pix_start = w_start.wcs_world2pix([[ra1, dec1]], 1)
        ra_dec_1_pix_end = w_end.wcs_world2pix([[ra1, dec1]], 1)

        print(f"reference WCS end-transformed ra1, dec1: {ra_dec_1_pix_end[0][0]}, {ra_dec_1_pix_end[0][1]}")
        print(f"reference truth x1, y1:                 {x1}, {y1}")
        print("*****************************************************************************")
        print(f"reference WCS ra0, dec0:                {ra_dec_0_pix[0][0]}, {ra_dec_0_pix[0][1]}")
        print(f"reference truth x0, y0:                 {x0}, {y0}")

        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap="gray", norm=ImageNormalize(img, interval=ZScaleInterval()))
        ax.invert_yaxis()
        ax.set_title(filename)

        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()


def check_times(path):
    """
    Check time consistency across image headers, reference WCS headers, and truth tables.

    Parameters
    ----------
    path : str
        Output directory containing ``images`` and ``reference`` subdirectories.
    """
    image_dir = os.path.join(path, "images")
    reference_dir = os.path.join(path, "reference")

    files = sorted(glob.glob(os.path.join(image_dir, "????.fits")))
    truth_hdul = fits.open(os.path.join(reference_dir, "reference_truth.fits"))

    for filename in files:
        image_id = image_id_from_filename(filename)

        img_hdul = fits.open(filename)
        truth_wcs_hdul = fits.open(os.path.join(reference_dir, f"{image_id}.wcs.fits"))

        wcs_start_time = truth_wcs_hdul[0].header["DATE-BEG"]
        wcs_end_time = truth_wcs_hdul[0].header["DATE-END"]

        t_start = img_hdul[0].header["DATE-BEG"]
        t_end = img_hdul[0].header["DATE-END"]

        assert wcs_start_time == t_start, f"Start time mismatch for image {image_id}"
        assert wcs_end_time == t_end, f"End time mismatch for image {image_id}"

        sat_hdu_name = f"SAT_{image_id}"
        if "t" in truth_hdul[sat_hdu_name].columns.names:
            truth_time = Time(truth_hdul[sat_hdu_name].data.t[0], format="gps").tai.isot
            assert wcs_start_time == truth_time, f"Truth time mismatch for image {image_id}"


def check_num_images(path):
    """
    Report the number of generated images.

    Parameters
    ----------
    path : str
        Output directory containing an ``images`` subdirectory.
    """
    image_dir = os.path.join(path, "images")
    files = sorted(glob.glob(os.path.join(image_dir, "????.fits")))

    print(f"Total images: {len(files)}")

    plt.figure()
    plt.hist(np.arange(len(files)))
    plt.title("Generated image count")
    plt.savefig(os.path.join("tests", "num_images.png"))
    plt.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--num-img", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    path = config["outdir"]

    check_endpoints(path, num_img=args.num_img, verbose=args.verbose)
    check_times(path)
    check_num_images(path)
