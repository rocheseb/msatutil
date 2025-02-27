import os
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from msatutil.msat_interface import msat_collection
from msatutil.msat_targets import get_target_dict


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
        latlon_step=0.5,
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
    l2.heatmap(
        "xch4",
        latlon=True,
        ax=ax[0],
        colorbar_label="XCH$_4$ (ppb)",
        vmin=1850,
        vmax=2100,
        over=None,
        under=None,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=0.5,
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
        latlon_step=0.5,
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
    l3.heatmap(
        "xch4",
        latlon=True,
        ax=ax[0],
        colorbar_label="XCH$_4$ (ppb)",
        vmin=1850,
        vmax=2100,
        over=None,
        under=None,
        latlon_padding=0.2,
        lab_prec=1,
        latlon_step=0.5,
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
        latlon_step=0.5,
        cb_fraction=0.03,
        cmap="Greys_r",
    )
    ax[0].set_title(title, fontsize=7)
    ax[1].set_title("")
    ax[1].set_ylabel("")
    fig.savefig(outfile, bbox_inches="tight", dpi=dpi, transparent=False)
    plt.close(fig)
    l3.close()


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
        "--overwrite",
        action="store_true",
        help="if given, remake all images even if they already exist",
    )
    args = parser.parse_args()

    td = get_target_dict(args.file_list)
    for t in td:
        for c in td[t]:
            for p in td[t][c]:
                td[t][c][p] = Path(td[t][c][p])
    first_value = list(next(iter(next(iter(td.values())).values())).values())[0].name

    if "_L1B_" in first_value:
        plot_func = plot_l1
    elif "_L2_" in first_value:
        plot_func = plot_l2
    elif "_L3_" in first_value:
        plot_func = plot_l3
    else:
        raise Exception("Couldn't recognize the file type")

    # loop over bucket files, download the data file, make the png plot, upload the png, remove the downloaded data file
    for t in td:
        for c in td[t]:
            for p in td[t][c]:
                gs_file = td[t][c][p]
                png_file = (
                    Path(args.out_dir) / f"{gs_file.parts[3]}_{gs_file.name.replace('.nc','.png')}"
                )
                if not args.overwrite and png_file.exists():
                    continue
                try:
                    downloaded_file = Path(args.download_dir) / gs_file.name
                    os.system(f"gsutil cp {str(gs_file).replace('gs:/','gs://')} {downloaded_file}")
                    plot_func(
                        str(downloaded_file),
                        png_file,
                        title=os.path.splitext(png_file.name)[0],
                    )
                except Exception:
                    print(f"Could not make the plot for {gs_file}")
                    continue
                finally:
                    if downloaded_file.exists():
                        os.remove(downloaded_file)

    # upload the pngs to the given bucket
    if args.overwrite:
        os.system(f"gsutil -m cp {args.out_dir}/*.png {args.bucket}/")
    else:
        os.system(f"gsutil -m cp -n {args.out_dir}/*.png {args.bucket}/")


if __name__ == "__main__":
    main()
