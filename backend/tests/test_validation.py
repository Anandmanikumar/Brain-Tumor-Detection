
from pathlib import Path

from app.services.validation import (
    inspect_nifti,
    validate_matching_geometry,
)


# Replace these with actual files on your computer.
MRI_PATH = Path(r"C:\Users\HP\BraTS2021_00000_flair.nii.gz")
MASK_PATH = Path(r"C:\Users\HP\00000.nii.gz")


def main():
    print("\n--- MRI INFORMATION ---")

    mri = inspect_nifti(MRI_PATH)
    print(mri)

    print("\n--- MASK COMPATIBILITY ---")

    result = validate_matching_geometry(
        MRI_PATH,
        MASK_PATH,
    )

    print(result)


if __name__ == "__main__":
    main()
