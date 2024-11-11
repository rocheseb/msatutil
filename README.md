# MSATutil

Utility programs for reading **MethaneSAT/AIR** files and making quick diagnostics plots

# Install

`git clone https://github.com/rocheseb/msatutil`

`pip install -e msatutil`

To be able to run the [mair_geoviews](notebooks/mair_geoviews.ipynb) notebook use this command instead:

`pip install -e msatutil[notebooks]`

For loading csvs from a GCS filepath:

`pip install -e msatutil[gcsfs]`

# msatutil

## msatutil.msat_dset

Just extends netCDF4.Dataset to allow opening files in the google cloud storage starting with **gs://**

### msatutil.msat_nc

The **msat_nc** class represents a single L1/L2/L2-pp/L3 file with added metadata and convenience methods for browsing the data

Most useful methods are:

* **msat_nc.show_all**: show all the variables in the file and their dimensions
* **msat_nc.search**: search for a given keyword amongst the variables in the file
* **msat_nc.fetch**: fetch the first variable data that corresponds to the given keyword
* **msat_nc.show_sv**: for L2 files, show the state vector metadata
* **msat_nc.fetch_varpath**: get the full variable path that corresponds to the given keyword

#### msatutil.msat_interface

The most important object here is the **msat_collection** class which can open a list of L1/L2/L3 files

Its most useful methods are:

* **msat_collection.pmesh_prep**: returns a given variable from all the files concatenated along-track

* **msat_collection.grid_prep**: returns rough L3, the given variable on a regular lat/lon grid using mean aggregation for overlaps

* **msat_collection.heatmap**: use matplotlib's pcolormesh to plot the given variable either in along-/across-track indices or in lat/lon

* **msat_collection.hist**: makes a histogram of the given variable

It has most of the **msat_nc** convenience methods

There is a **get_msat** function to generate a **msat_collection** from all the files in a directory


### msatutil.compare_heatmaps

Can be used to compare granules from two different **msat_collection** objects

### msatutil.make_spatial_index

From a csv of filespaths, generate ESRI shapefile and geojson showing data coverage for L2pp or L3 files. There is an option for simplifying the output polygons to create small files. Currently, this function is very slow for L2 files due to the overhead of loading and re-gridding.

It interfaces with [mair_geoviews.py](msatutil/mair_geoviews.py)

Usage:

`python make_spatial_index.py -c gs://path/to/L2_pp.csv --working_dir . --load_from_chkpt FALSE --save_frequency 2 --out_path l2_test --l2_data`

Check detailed usage info with

`python make_spatial_index.py -h`

# Notebooks

There are notebooks showing example usage of the msatutil programs:

[msat_interface](notebooks/msat_interface_example.ipynb)

[compare_heatmaps](notebooks/compare_heatmaps_example.ipynb)

[mair_geoviews](notebooks/mair_geoviews.ipynb)

[mair_spatial_index](notebooks/mair_spatial_index.ipynb)

### Running mair_geoviews.ipynb

[mair_geoviews](notebooks/mair_geoviews.ipynb) is for plotting L3 data or a full flight worth of L1/L2/L2-pp data using Holoviz libraries

#### with a local webserver

e.g. from the parent directory of the cloned msatutil repo:

`panel serve --show msatutil/notebooks/mair_geoviews.ipynb`

Or using the **mairhtml** console script with a direct file path and the **--serve** argument:

`mairhtml l3_file_path out_path --serve`

#### with jupyter

`jupyter notebook msatutil/notebooks/mair_geoviews.ipynb`


# Console scripts

**mairls** is a console entry point for listing and subsetting MethaneAIR files from csv tables of metadata

It interfaces with [mair_ls.py](msatutil/mair_ls.py)

Check detailed usage info with

`mairls -h`

**mairhtml** is a console entry point for generating static html plot of L3 mosaic files.

It interfaces with [mair_geoviews.py](msatutil/mair_geoviews.py)

Usage:

`mairhtml in_path out_path`

Check detailed usage info with

`mairhtml -h`

**in_path** can be the path to a directory or file, it accepts **gs://** paths

The following directory structure is assumed when **in_path** is a directory:

```
in_path
├── target_dir
│   ├── resolution_dir
│   │   └── mosaic_file.nc
│   └── resolution_dir2
│       └── mosaic_file.nc
└── target_dir2
    ├── resolution_dir
    │   └── mosaic_file.nc
    └── resolution_dir2
        └── mosaic_file.nc
```

When **in_path** is a direct path to a file, **mairhtml** can be given the **--serve** argument to popup the plot with a local webserver (with live regridding)

**in_path** can also be a **.csv** file with metadata on MethaneAIR files, in that case all the arguments of **mairls** can be used with **mairhtml** and the code will

generate static maps for all the files matching the mairls arguments.

Calling **mairhtml** with a **.csv** file and the **--show** argument won't generate plots, it is the same as calling **mairls**


#### Other

[generate_html_index.sh](msatutil/generate_html_index.sh) can be used to add index.html files under a directory structure

It needs **[tree](https://linux.die.net/man/1/tree)** and is called with **mairhtml** when the **-i/--index** argument is given

Usage:

`sh generate_html_index.sh path_to_directory`


#### Contact

sroche@g.harvard.edu

