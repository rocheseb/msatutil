import os
import re
import subprocess
from setuptools import setup, find_packages


_mydir = os.path.dirname(__file__)

# parse version number from msatutil/__init__.py
version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
with open(os.path.join(_mydir, "msatutil", "__init__.py")) as f:
    content = f.read()
re_search = re.search(version_re, content, re.M)
if re_search:
    version_str = re_search.group(1)
else:
    raise RuntimeError("Could not parse version string from __init__.py")

# add the git hash to version_str
if os.path.exists(os.path.join(_mydir, ".git")):
    git_hash = subprocess.check_output(
        "git rev-parse --verify --short HEAD", cwd=_mydir, text=True, shell=True
    ).strip()
    git_version_str = f"v{version_str}"
    tags = subprocess.check_output("git tag", cwd=_mydir, text=True, shell=True)
    if git_version_str not in tags:
        subprocess.check_output(
            f"git tag -a {git_version_str} {git_hash} -m 'tagged by setup.py to {version_str}'",
            cwd=_mydir,
            text=True,
            shell=True,
        )
    version_str = f"{version_str}+git.{git_hash}"

# extra dependencies for mair_geoviews.ipynb and loading csv straight from google cloud in make_spatial_index.py
extras = {
    "notebooks": [
        "notebook",
        "panel",
        "holoviews",
        "cartopy",
        "geoviews",
        "datashader",
        "gcsfs",
    ],
}

setup(
    name="msatutil",
    description="Utility codes to read and plot from MethaneSAT/AIR files",
    author="Sebastien Roche",
    author_email="sroche@g.harvard.edu",
    version=version_str,
    url="https://github.com/rocheseb/msatutil",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "dask",
        "netcdf4",
        "matplotlib",
        "pandas",
        "scipy",
        "tqdm",
        "google-cloud-storage",
        "pyogrio"
    ],
    extras_require=extras,
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.9,<3.12",
    entry_points={
        "console_scripts": [
            "mairl3html=msatutil.mair_geoviews:main",
            "mairls=msatutil.mair_ls:main",
        ],
    },
)
