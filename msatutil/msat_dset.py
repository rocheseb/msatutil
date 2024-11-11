import os
import re
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.client import Client
from netCDF4 import Dataset
from typing import Union


class msat_dset(Dataset):
    """
    Class to open a netcdf file on a google cloud bucket
    Download the file in memory and produce a netCDF4._netCDF4.Dataset object
    """

    def __init__(self, nc_target: Union[str, Blob], client: Client = None):
        """
        nc_target can be:
            - a netcdf file path (str)
            - a google cloud file path starting with gs:// (str)
            - a google cloud blob (Blob)

        client (Client): used when nc_target is a blob, defaults to storage.Client()
        """
        if "blob" in str(type(nc_target)).lower() or nc_target.startswith("gs://"):
            if nc_target.startswith("gs://"):
                if client is None:
                    client = storage.Client()
                nc_target = Blob.from_string(nc_target, client=client)
            with nc_target.open("rb") as gcloud_file:
                data = gcloud_file.read()
            filename = "memory"
        else:
            data = None
            filename = nc_target

        super().__init__(filename, memory=data)

        # Can't simply do self.source = nc_target
        # that would be trying to write to the read-only Dataset parameters
        if "blob" in str(type(nc_target)).lower():
            self.__dict__["source"] = f"gs://{nc_target.bucket.name}/{nc_target.name}"
        else:
            self.__dict__["source"] = nc_target

        @property
        def source(self):
            return self.__dict__["source"]


class cloud_file:
    def __init__(self, file_path: str, client=None):
        self.file_path = file_path
        self.client = client

    def __enter__(self):
        if self.file_path.startswith("gs://"):
            if self.client is None:
                client = storage.Client()
            target = Blob.from_string(self.file_path, client=client)
            self.file = target.open("rb")
        else:
            self.file = open(self.file_path, "rb")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.file.closed:
            self.file.close()


def gs_list(gs_path: str, srchstr=None) -> list:
    """
    Return a list of blobs under gs_path

    gs_path (str): google storage path
    srchstr (Optional[str]): file search pattern (accepts wildcards *)
    """
    input_dir_blob = Blob.from_string(gs_path)
    bucket_name = input_dir_blob.bucket.name
    prefix = input_dir_blob.name
    flist = list(storage.Client().list_blobs(bucket_name, prefix=prefix))
    if srchstr is not None:
        srchstr = srchstr.replace("*", "(.*?)")
        flist = [i for i in flist if re.match(srchstr, os.path.basename(i.name))]
    return [f"gs://{bucket_name}/{i.name}" for i in flist]
