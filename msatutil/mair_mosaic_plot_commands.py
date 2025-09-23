import argparse
import os
from pathlib import Path

from msatutil.mair_targets import get_target_dict
from msatutil.msat_targets import gs_posixpath_to_str


def make_commands(file_list: str, command_file: str, out_dir: str):
    td = get_target_dict(file_list)

    commands = []
    for c in td:
        for f in td[c]:
            for p in td[c][f]:
                a = list(td[c][f][p].keys())[0]
                l3_dir = Path(td[c][f][p][a]).parent.parent.parent
                commands += [
                    f"mairhtml {gs_posixpath_to_str(l3_dir)} {os.path.join(out_dir,c,f)} --title {f} --tab-title {f} -v xch4 --pixel-resolution 50 50\n"
                ]
    with open(command_file, "w") as f:
        f.writelines(commands)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_list", help="full path to file listing L3 mosaic paths")
    parser.add_argument(
        "-c",
        "--command-file",
        default="make_mair_mosaic_plots.sh",
        help="output commands file",
    )
    parser.add_argument("-o", "--out-dir", help="directory under which the plots will be saved")
    args = parser.parse_args()

    make_commands(args.file_list, args.command_file, args.out_dir)


if __name__ == "__main__":
    main()
