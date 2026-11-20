
import uuid

from app.services.storage import (
    create_job_directory,
    get_job_directory,
    validate_nifti_filename,
    delete_job_files,
    StorageError,
)


def main():
    job_id = str(uuid.uuid4())

    print("Creating job directory...")

    directories = create_job_directory(job_id)

    print(directories)

    print("\nTesting filename validation...")

    assert validate_nifti_filename(
        "BraTS2021_00000_flair.nii.gz"
    ) == "BraTS2021_00000_flair.nii.gz"

    try:
        validate_nifti_filename("malware.exe")
    except StorageError:
        print("Invalid file extension correctly rejected.")

    print("\nTesting job retrieval...")

    retrieved = get_job_directory(job_id)

    assert retrieved["job_dir"] == directories["job_dir"]

    print("Job retrieval successful.")

    print("\nTesting cleanup...")

    delete_job_files(job_id)

    print("Job files deleted successfully.")

    print("\nALL STORAGE TESTS PASSED")


if __name__ == "__main__":
    main()
