import argparse
import mimetypes
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print(
        "Google drive API not installed, reinstall msatutil with python -m pip install -e .[gdrive]"
    )
    service_account = None


def upload_file(
    outfile: str,
    service_account_file: str,
    folder_id: str,
    overwrite: bool = False,
) -> str:
    """
    Upload outfile to a google drive folder

    Inputs:
        outfile (str): full path to file to upload
        service_account_file (str): full path to the Google Drive API service account file
        folder_id (str): Google Drive folder ID, must have been shared with the service account
        overwrite (bool): if True, overwrite files with the same name
    Outputs:
        (str): direct link to the uploaded file
    """
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )

    drive_service = build("drive", "v3", credentials=credentials)

    filename = os.path.basename(outfile)

    # check if file already exists
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    results = (
        drive_service.files()
        .list(
            q=query,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="allDrives",
        )
        .execute()
    )
    files = results.get("files", [])

    # If the file already exists, return the file ID
    if files and not overwrite:
        return f"https://drive.google.com/uc?export=view&id={files[0]['id']}"

    file_metadata = {
        "name": filename,
        "parents": [folder_id],
    }

    mimetype, _ = mimetypes.guess_type(outfile)
    if mimetype is None:
        mimetype = "plain/text"

    media = MediaFileUpload(outfile, mimetype=mimetype, resumable=True)

    if files:
        file_id = files[0]["id"]
        drive_service.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True,
        ).execute()
    else:
        uploaded_file = (
            drive_service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        file_id = uploaded_file["id"]

    return f"https://drive.google.com/uc?export=view&id={file_id}"


def trash_file(service_account_file: str, folder_id: str, filename: str) -> bool:
    """
    Trash a file from a specified Google Drive folder by its name.

    Inputs:
        service_account_file (str): path to the service account JSON key file
        folder_id (str): ID of the Google Drive folder
        filename (str): name of the file to delete

    Returns:
        bool: True if the file was trashed, False if not found
    """
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    drive_service = build("drive", "v3", credentials=credentials)

    # Search for the file in the given folder
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    results = (
        drive_service.files()
        .list(
            q=query,
            fields="files(id, name, trashed)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="allDrives",
        )
        .execute()
    )

    files = results.get("files", [])

    if not files or files[0].get("trashed"):
        return False  # file not found or already trashed

    file_id = files[0]["id"]

    # Delete the file
    drive_service.files().update(
        fileId=file_id, body={"trashed": True}, supportsAllDrives=True
    ).execute()

    return True


def cleanup(service_account_file: str, folder_id: str, local_folder: str, contains: str):
    """
    Trash drive files under folder_id if they are not under local_folder

    Inputs:
        service_account_file (str): path to the service account JSON key file
        folder_id (str): ID of the Google Drive folder
        local_folder (str): full path to local directory the drive will match
        contains (str): key included in files to delete
    """
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    drive_service = build("drive", "v3", credentials=credentials)

    # Search for the file in the given folder
    query = f"'{folder_id}' in parents and name contains '{contains}' and trashed = false"

    files = []
    page_token = None

    while True:
        response = (
            drive_service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, trashed)",
                pageSize=1000,  # max per request
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora="allDrives",
                pageToken=page_token,  # fetch next page if available
            )
            .execute()
        )

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    drive_files = [i["name"] for i in files if not i["trashed"]]
    local_files = [i.name for i in Path(local_folder).glob("*.png")]
    dif = set(drive_files).difference(local_files)
    for i in tqdm(dif):
        trash_file(service_account_file, folder_id, i)


def get_file_link(service_account_file: str, folder_id: str, filename: str) -> Optional[str]:
    """
    Retrieve the file ID of a file in a specified Google Drive folder given its filename.

    Inputs:
        filename (str): the name of the file to look for in the given drive folder
    Outputs:
        (Optional[str]): direct Google Drive link to the file
    """
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )

    drive_service = build("drive", "v3", credentials=credentials)

    # Query to find the file by its name and the folder it belongs to
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"

    # Perform the search
    results = (
        drive_service.files()
        .list(
            q=query,
            fields="files(id, name, trashed)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="allDrives",
        )
        .execute()
    )

    # Get the first matching file (if any)
    files = results.get("files", [])

    if files and not files[0].get("trashed"):
        return f"https://drive.google.com/uc?export=view&id={files[0]['id']}"
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in-dir",
        help="full path to folder with files to upload",
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
        "-o",
        "--overwrite",
        action="store_true",
        help="if given, overwrite existing files in the Google Drive",
    )
    args = parser.parse_args()

    file_list = [os.path.join(args.in_dir, i) for i in os.listdir(args.in_dir)]
    for outfile in file_list:
        link = upload_file(
            outfile,
            args.service_account_file,
            args.google_drive_id,
            args.overwrite,
        )
        if link is not None:
            print(f"Uploaded {outfile}")


if __name__ == "__main__":
    main()
