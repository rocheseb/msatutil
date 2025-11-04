import argparse
import gc
import os
from pathlib import Path

import holoviews as hv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer

from msatutil.mair_geoviews import save_static_plot_with_widgets, show_map
from msatutil.msat_gdrive import upload_file as google_drive_upload
from msatutil.msat_interface import msat_collection
from msatutil.msat_targets import get_target_dict, gs_posixpath_to_str


def qaqc_filter(qaqc_file) -> bool:
    """
    Filter some collects based on some L2 qaqc metrics

    Inputs:
        qaqc_file (str): L2 qaqc csv file
    Outputs:
        (str): empty string if the collect passes, reason for failing returned otherwise
    """
    data = pd.read_csv(qaqc_file, names=["var", "status", "value"], skiprows=1)
    if any(data["status"] == "fail"):
        return "Fail"

    # Don't include scenes with more than 5% missing frames
    missing_frames_fraction = float(
        data.loc[data["var"] == "Missing frames fraction"].iloc[0].value
    )
    if missing_frames_fraction > 0.05:
        return ">5% Missing frames"

    # Don't include scenes with less than 290 frames
    number_of_frames = float(data.loc[data["var"] == "Number of frames"].iloc[0].value)
    if number_of_frames < 290:
        return "<290 frames"

    # Don't include scenes more than 70% flagged
    flag_fraction = float(data.loc[data["var"] == "CH4 flagged fraction"].iloc[0].value)
    if flag_fraction > 0.7:
        return ">70% flagged"

    # Don't include scenes where the non-flagged XCH4 standard deviation > 100 ppb
    xch4_std = float(data.loc[data["var"] == "XCH4 flag0 STD"].iloc[0].value)
    if xch4_std > 100:
        return ">100ppb XCH4 STD"

    # Don't include scenes with anomalous delta_pressure that could be contaminated by aerosols
    anomalous_o2dp = data.loc[data["var"] == "Anomalous O2DP"].iloc[0].value == "true"
    if anomalous_o2dp:
        return "Anomalous O2DP"

    # Don't include scenes more than 60% cloudy/shadowy
    cloud_fraction = float(data.loc[data["var"] == "Cloud fraction"].iloc[0].value)
    shadow_fraction = float(data.loc[data["var"] == "Shadow fraction"].iloc[0].value)

    if cloud_fraction + shadow_fraction > 0.6:
        return ">60% clouds+shadows"

    return ""


def select_colorscale(mc: msat_collection) -> tuple[float, float]:
    """
    Get the vmin and vmax that will be used for the XCH4 plot
    """
    if mc.is_l2:
        xch4 = mc.pmesh_prep("product_co2proxy/xch4_co2proxy", use_valid_xtrack=True)
        if mc.use_dask:
            xch4 = xch4.compute()
    elif mc.is_l3:
        xch4 = mc.pmesh_prep("xch4")
        if mc.use_dask:
            xch4 = xch4.compute()
    if mc.is_postproc:
        flag = mc.pmesh_prep("product_co2proxy/main_quality_flag", use_valid_xtrack=True)
        if mc.use_dask:
            flag = flag.compute()
    else:
        flag = np.zeros(xch4.shape)

    med = np.nanmedian(xch4[flag == 0].filled(np.nan))
    q25, q75 = np.nanpercentile(xch4[flag == 0].filled(np.nan), [25, 75])
    std = 0.74 * (q75 - q25)

    STD_THRESHOLD = 65  # ppb
    if std > STD_THRESHOLD:
        vmin = med - 2 * STD_THRESHOLD
        vmax = med + 2 * STD_THRESHOLD
    else:
        vmin = med - 2 * std
        vmax = med + 2 * std

    return vmin, vmax


def plot_l1(l1_file, outfile, title="", add_basemap=False, dpi=300):
    matplotlib.use("Agg")
    plt.ioff()
    l1 = msat_collection([l1_file])
    fig, ax = plt.subplots(figsize=(6, 8), dpi=dpi, constrained_layout=True)
    ax.set_facecolor("black")
    l1.heatmap(
        "Band1/Radiance",
        latlon=True,
        ax=ax,
        colorbar_label="Radiance[1000]",
        extra_id=1000,
        extra_id_dim="spectral_channel",
        over=None,
        under=None,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=1,
        cb_fraction=0.03,
        add_basemap=add_basemap,
        cmap="Greys_r",
    )
    ax.set_title(title, fontsize=7)
    fig.savefig(outfile, bbox_inches="tight", dpi=dpi, transparent=False)
    plt.close(fig)
    l1.close()


def plot_l2(l2_file, outfile, title="", flagged=False, add_basemap=False, dpi=300):
    matplotlib.use("Agg")
    plt.ioff()
    l2 = msat_collection([l2_file])
    if flagged:
        flag = l2.pmesh_prep("product_co2proxy/main_quality_flag", use_valid_xtrack=True).compute()
    fig, ax = plt.subplots(1, 2, figsize=(11, 8), dpi=dpi, constrained_layout=True, sharey=True)
    ax[0].set_facecolor("black")
    vmin, vmax = select_colorscale(l2)
    l2.heatmap(
        "product_co2proxy/xch4_co2proxy",
        latlon=True,
        ax=ax[0],
        colorbar_label="XCH$_4$ (ppb)",
        vmin=vmin,
        vmax=vmax,
        over=None,
        under=None,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=1,
        cb_fraction=0.03,
        mask=None if not flagged else flag > 0,
        add_basemap=add_basemap,
    )
    l2.heatmap(
        "albedo",
        latlon=True,
        ax=ax[1],
        colorbar_label="Albedo",
        vmin=0,
        vmax=1,
        over=None,
        under=None,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=1,
        cb_fraction=0.03,
        cmap="Greys_r",
    )
    ax[0].set_title(title, fontsize=7)
    ax[1].set_title("")
    ax[1].set_ylabel("")
    fig.savefig(outfile, bbox_inches="tight", dpi=dpi, transparent=False)
    plt.close(fig)
    l2.close()


def plot_l3(l3_file, outfile, title="", add_basemap=False, dpi=300):
    matplotlib.use("Agg")
    plt.ioff()
    l3 = msat_collection([l3_file])
    fig, ax = plt.subplots(1, 2, figsize=(11, 8), dpi=dpi, sharey=True, constrained_layout=True)
    ax[0].set_facecolor("black")
    vmin, vmax = select_colorscale(l3)
    l3.heatmap(
        "xch4",
        latlon=True,
        ax=ax[0],
        colorbar_label="XCH$_4$ (ppb)",
        vmin=vmin,
        vmax=vmax,
        over=None,
        under=None,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=1,
        cb_fraction=0.03,
        add_basemap=add_basemap,
    )
    l3.heatmap(
        "albedo",
        latlon=True,
        ax=ax[1],
        colorbar_label="Albedo",
        vmin=0,
        vmax=1,
        over=None,
        under=None,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=1,
        cb_fraction=0.03,
        cmap="Greys_r",
    )
    ax[0].set_title(title, fontsize=7)
    ax[1].set_title("")
    ax[1].set_ylabel("")
    fig.savefig(outfile, bbox_inches="tight", dpi=dpi, transparent=False)
    plt.close(fig)
    l3.close()


def read_l4(l4_file: str):
    """
    Inputs:
        l4_file (str): full path to input L4_output_analysis file
    Outputs:
        lon (np.ndarray): array of longitudes
        lat (np.ndarray): array of latitudes
        flux (np.ndarray): array of CH4 mean flux (kg/hr)
        l3_file (str): full path to the corresponding L3 file
    """
    with Dataset(l4_file) as l4:
        x = l4["x"][:]
        y = l4["y"][:]
        flux = l4["mean_flux"][:]
        transformer = Transformer.from_crs(l4.utm_crs_code, "epsg:4326", always_xy=True)
        l3_root = str(Path(*Path(l4_file).parts[:4])).replace("4", "3")
        if "ProcessingMetadata" in l4.groups:
            l3_file = l4["ProcessingMetadata"].level3_product.replace("[prod]", l3_root)
        else:
            l3_file = None

    xx, yy = np.meshgrid(x, y, indexing="ij")

    lon, lat = transformer.transform(xx, yy)

    return lon, lat, flux, l3_file


def plot_l4(l4_file, outfile, title="", add_basemap=False, dpi=300):
    matplotlib.use("Agg")
    plt.ioff()

    lon, lat, flux, _ = read_l4(l4_file)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=dpi, sharey=True, constrained_layout=True)
    ax.set_facecolor("black")
    msat_collection.make_heatmap(
        ax,
        flux,
        lon=lon,
        lat=lat,
        xlabel="Longitude",
        ylabel="Latitude",
        colorbar_label="Mean flux (kg/hr)",
        over=None,
        under=None,
        add_basemap=add_basemap,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=1,
        cb_fraction=0.03,
        vmin=0,
        vmax=np.nanpercentile(flux, 97.5),
    )
    ax.set_title(title, fontsize=7)
    fig.savefig(outfile, bbox_inches="tight", dpi=dpi, transparent=False)
    plt.close(fig)


def plot_l4_html(l4_file, outfile, title="", width=550, height=450):
    """
    Make a mean CH4 flux html map from a L4 file
    """

    lon, lat, flux, l3_file = read_l4(l4_file)

    l4_plot = show_map(
        lon,
        lat,
        flux,
        width=width,
        height=height,
        cmap="viridis",
        clim=(0, np.nanpercentile(flux, 97.5)),
        title="Mean CH4 flux (kg/hr)",
    )

    if l3_file is not None:
        with msat_collection([l3_file], use_dask=False) as l3:
            vmin, vmax = select_colorscale(l3)
            lon = l3.pmesh_prep("lon")
            lat = l3.pmesh_prep("lat")
            xch4 = l3.pmesh_prep("xch4")
            albedo = l3.pmesh_prep("albedo")
        l3_plot_xch4 = show_map(
            lon,
            lat,
            xch4,
            width=width,
            height=height,
            cmap="viridis",
            clim=(vmin, vmax),
            title="XCH4 (ppb)",
            single_panel=True,
        )
        l3_plot_albedo = show_map(
            lon,
            lat,
            albedo,
            width=width,
            height=height,
            cmap="gray",
            clim=(np.nanmin(albedo), np.nanmax(albedo)),
            title="Albedo",
            single_panel=True,
        )

        plot = hv.Layout([l4_plot, l3_plot_xch4, l3_plot_albedo]).cols(2)
    else:
        plot = l4_plot

    if l4_file.startswith("/mnt/gcs/"):
        l4_file = l4_file.replace("/mnt/gcs/", "gs://")

    save_static_plot_with_widgets(
        outfile,
        plot,
        cmap="viridis",
        layout_title="MethaneSAT L4 CORE",
        layout_details=l4_file,
        browser_tab_title="MethaneSAT L4",
    )

    del lat, lon, xch4, albedo, flux, l4_plot, plot
    if l3_file is not None:
        del l3_plot_xch4, l3_plot_albedo
    
    gc.collect()


def download_file(download_dir: str, p: Path, use_mount: bool) -> Path:
    """
    Inputs:
        download_dir (str): full path to the directory where file will be downloaded
        p (Path): file to download
        use_mount (bool): if True, just return p
    Outputs:
        downloaded_file (Path): output file path
    """
    if use_mount:
        return p

    downloaded_file = Path(download_dir) / p.name

    if not downloaded_file.exists():
        os.system(f"gsutil cp {gs_posixpath_to_str(p)} {downloaded_file}")

    return downloaded_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file-list",
        required=True,
        help="input file with list of bucket paths",
    )
    parser.add_argument(
        "-d",
        "--download-dir",
        required=True,
        help="full path to the directory where data file will be downloaded",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="full path to directory where the PNGs will be saved",
    )
    parser.add_argument(
        "-b",
        "--bucket",
        help="upload bucket path (starts with gs://)",
    )
    parser.add_argument(
        "-s",
        "--service-account-file",
        help="full path to the Google service account json file",
    )
    parser.add_argument(
        "-g",
        "--google-drive-id",
        help="Google Drive ID of the upload folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="if given, remake all images even if they already exist",
    )
    parser.add_argument(
        "--qaqc-list",
        default=None,
        help="if given, also pull the L2 QAQC csv files to filter out some scenes",
    )
    parser.add_argument(
        "--qaqc-filter",
        default=None,
        help="full path to a csv file listing collects to exclude",
    )
    parser.add_argument(
        "--log-file",
        default="msat_images.log",
        help="Full path to the output log file listing collection that pass/fail the image generation",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="if given, make html maps",
    )
    parser.add_argument(
        "--use-mount", action="store_true", help="if given, use /mnt/gcs/ instead of gs://"
    )
    args = parser.parse_args()

    td = get_target_dict(args.file_list)
    if args.qaqc_list:
        qcd = get_target_dict(args.qaqc_list)
    for t in td:
        for c in td[t]:
            for p in td[t][c]:
                if args.use_mount:
                    td[t][c][p] = td[t][c][p].replace("gs://", "/mnt/gcs/")
                td[t][c][p] = Path(td[t][c][p])
                if args.qaqc_list:
                    if args.use_mount:
                        qcd[t][c][p] = qcd[t][c][p].replace("gs://", "/mnt/gcs/")
                    qcd[t][c][p] = Path(qcd[t][c][p])
    first_value = list(next(iter(next(iter(td.values())).values())).values())[0].name

    if "_L1B_" in first_value:
        plot_func = plot_l1
    elif "_L2_" in first_value:
        plot_func = plot_l2
    elif "_L3_" in first_value:
        plot_func = plot_l3
    elif "_L4_" in first_value:
        plot_func = plot_l4_html if args.html else plot_l4
    else:
        raise Exception("Couldn't recognize the file type")

    if args.qaqc_filter:
        filter_df = pd.read_csv(
            args.qaqc_filter,
            names=["target", "collection", "reason"],
            dtype={"target": int, "collection": str, "reason": "str"},
            skiprows=1,
        )
        filter_targets = filter_df[filter_df["collection"].isna()]["target"].values

    extension = ".html" if args.html else ".png"
    log = pd.DataFrame(columns=["target_ID", "collection_ID", "processing_ID", "status"])
    # loop over bucket files, download the data file, make the png plot, upload the png, remove the downloaded data file
    for t in td:
        for c in td[t]:
            for p in td[t][c]:
                gs_file = td[t][c][p]
                output_file = (
                    Path(args.out_dir)
                    / f"{gs_file.parts[5 if args.use_mount else 3]}_{gs_file.name.replace('.nc',extension)}"
                )
                if args.qaqc_list and (t not in qcd or c not in qcd[t] or p not in qcd[t][c]):
                    print(f"No L2 qaqc corresponding to: {gs_file}")
                    log.loc[len(log)] = [t, c, p, "No L2 qaqc file found"]
                    continue
                if args.qaqc_filter:  # "manual" filter
                    if t in filter_targets or c in filter_df["collection"].values:
                        print(f"Filtered out (manual): {gs_file}")
                        if c in filter_df["collection"].values:
                            reason = filter_df.loc[filter_df["collection"] == c]["reason"].values[0]
                        else:
                            reason = filter_df.loc[filter_df["target"] == t]["reason"].values[0]
                        log.loc[len(log)] = [t, c, p, f"Excluded by manual review: {reason}"]
                        output_file.unlink(missing_ok=True)
                        continue
                if args.qaqc_list:  # automated filter
                    qc_gs_file = qcd[t][c][p]
                    downloaded_qc_file = download_file(args.download_dir, qc_gs_file)
                    skip_plot = qaqc_filter(downloaded_qc_file)
                    if skip_plot:
                        print(f"Filtered out (auto) for {skip_plot}: {gs_file}")
                        log.loc[len(log)] = [t, c, p, f"Excluded by automated filter: {skip_plot}"]
                        output_file.unlink(missing_ok=True)
                        continue
                if not args.overwrite and output_file.exists():
                    log.loc[len(log)] = [t, c, p, "pass"]
                    continue
                try:
                    downloaded_file = download_file(args.download_dir, gs_file, args.use_mount)
                    plot_func(
                        str(downloaded_file),
                        output_file,
                        title=os.path.splitext(output_file.name)[0],
                    )
                except Exception as e:
                    print(f"Could not make the plot for {gs_file}", e)
                    log.loc[len(log)] = [t, c, p, "Cloud not make the plot"]
                    continue
                finally:
                    if not args.use_mount:
                        downloaded_file.unlink(missing_ok=True)
                    gc.collect()
                log.loc[len(log)] = [t, c, p, "pass"]
    log.to_csv(args.log_file, index=False)

    # upload the pngs to the given bucket
    if args.overwrite:
        os.system(f"gsutil -m cp {args.out_dir}/*{extension} {args.bucket}/")
    else:
        os.system(f"gsutil -m cp -n {args.out_dir}/*{extension} {args.bucket}/")

    if args.google_drive_id is not None and args.service_account_file is not None:
        outfile_list = [os.path.join(args.out_dir, i) for i in os.listdir(args.out_dir)]
        for outfile in outfile_list:
            google_drive_upload(
                outfile,
                args.service_account_file,
                args.google_drive_id,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
