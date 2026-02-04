"""Parameters and common functions to be used in notebook-based analysis workflow."""

from pathlib import Path

from cloudpathlib import AnyPath
from zipfile import ZipFile
import os

from pyCIAM import __file__

from dask_gateway import GatewayCluster


SLIIDERS_VERS = "v1.2"
RES_VERS = "regional-v0.1"

# Output dataset attrs
HISTORY = """regional-v0.1: First version associated with Regional SCC manuscript."""
AUTHOR = "Ian Bolliger"
CONTACT = "ian@reask.earth"

##############
# PYCIAM SPECS
##############
SEG_CHUNKSIZE = 1

##################
# ROOT DIRECTORIES
##################
DIR_SCRATCH = AnyPath("gs://impactlab-data-scratch/dscim-coastal-tmp")
DIR_COASTAL = AnyPath("gs://impactlab-data/coastal")
DIR_PROJ = DIR_COASTAL / "local-scc-model"
DIR_DATA_EXT = DIR_COASTAL / "data"
DIR_INT_EXT = DIR_DATA_EXT / "int"
DIR_RAW_EXT = DIR_DATA_EXT / "raw"
DIR_DATA = DIR_PROJ / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_INT = DIR_DATA / "int"
DIR_RES = DIR_PROJ / "results"

##################
# MODEL PARAMS
##################

PATH_PARAMS = Path("params.json")

##################
# SOCIOECON INPUTS
##################

# SLIIDERS
PATH_SLIIDERS_OLD = DIR_RAW_EXT / "ciam_inputs/pyCIAM_inputs_v6_6_0.zarr"
PATH_SLIIDERS = DIR_INT / "sliiders-ir.zarr"
PATH_SLIIDERS_SEG = DIR_SCRATCH / "sliiders-seg.zarr"

#####
# SLR
#####
PATH_SLR_RAW = (
    DIR_INT_EXT / "sea_levels/sea_level_rise_projections/all-workflows-all-ssps-v0.1.zarr"
)
DIR_SLR_GMSL_RAW = DIR_RAW_EXT / "slr/ar6/ar6/global/full_sample_workflows"
PATH_SLR_INT = DIR_SCRATCH / "ar6-lsl.zarr"
BIN_MAP_PATH = DIR_INT / "slr-bin-map-path.zarr"

###########################
# PYCIAM INTERMEDIATE FILES
###########################

DIR_SURGE_LOOKUP = DIR_INT / "surge-lookup"
PATHS_SURGE_LOOKUP = {
    "seg": DIR_SURGE_LOOKUP / "surge-lookup-seg.zarr",
    "seg_ir": DIR_SURGE_LOOKUP / "surge-lookup-seg-ir.zarr",
}

PATH_REFA = DIR_INT / "refa.zarr"

#########
# OUTPUTS
#########
TMPPATH = DIR_SCRATCH / "pyciam-output-tmp.zarr"
PATH_OUTPUT = DIR_RES / "pyciam-output.zarr"


#######
# FUNCS
#######
def start_dask_cluster(profile="micro", **kwargs):
    cluster = GatewayCluster(profile=profile, **kwargs)
    client = cluster.get_client()
    upload_pyciam(client)
    return client, cluster


def _zipdir(
    path,
    zip_filename,
    skip_files=(
        ".git",
        ".github",
        ".pytest_cache",
        "tests",
        "docs",
        "deploy",
        "notebooks",
        ".ipynb_checkpoints",
        "__pycache__",
        ".coverage",
        "dockerignore",
        ".gitignore",
        ".gitlab-ci.yml",
        ".gitmodules",
        "pyclaw.log",
        "run_tests.sh",
    ),
):
    with ZipFile(zip_filename, "w") as ziph:
        for root, dirs, files in os.walk(path):
            for file in files:
                if any([f in file.split("/") for f in skip_files]):
                    continue
                # Create a relative path for files to preserve the directory structure
                # within the ZIP archive. This relative path is based on the directory
                # being zipped, so files are stored in the same structure.
                relative_path = os.path.relpath(
                    os.path.join(root, file), os.path.join(path, "..")
                )
                ziph.write(os.path.join(root, file), arcname=relative_path)


def upload_pyciam(client):
    package_dir = Path(__file__).parent
    zip_filename = "/tmp/pyCIAM.zip"  # Output ZIP file name
    _zipdir(package_dir, zip_filename)
    client.upload_file(zip_filename)