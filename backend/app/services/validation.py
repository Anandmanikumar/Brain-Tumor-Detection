
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np


class NiftiValidationError(ValueError):
    """Raised when a NIfTI file is invalid or incompatible."""
    pass


def inspect_nifti(file_path: str | Path) -> Dict:
    """
    Inspect a 3D NIfTI file without loading the complete volume
    into memory.
    """

    path = Path(file_path)

    if not path.exists():
        raise NiftiValidationError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise NiftiValidationError(
            f"Expected a file, received: {path}"
        )

    if not (
        path.name.endswith(".nii")
        or path.name.endswith(".nii.gz")
    ):
        raise NiftiValidationError(
            "Only .nii and .nii.gz files are supported."
        )

    if path.stat().st_size == 0:
        raise NiftiValidationError(
            "NIfTI file is empty."
        )

    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise NiftiValidationError(
            f"Unable to read NIfTI file: {exc}"
        ) from exc

    shape = image.shape

    if len(shape) != 3:
        raise NiftiValidationError(
            f"Expected a 3D MRI volume, received shape {shape}."
        )

    if any(dimension <= 0 for dimension in shape):
        raise NiftiValidationError(
            f"Invalid image dimensions: {shape}"
        )

    affine = image.affine

    if affine.shape != (4, 4):
        raise NiftiValidationError(
            "Invalid NIfTI affine matrix."
        )

    if not np.isfinite(affine).all():
        raise NiftiValidationError(
            "Affine matrix contains non-finite values."
        )

    voxel_spacing = image.header.get_zooms()[:3]

    if (
        len(voxel_spacing) != 3
        or not np.isfinite(voxel_spacing).all()
        or any(spacing <= 0 for spacing in voxel_spacing)
    ):
        raise NiftiValidationError(
            f"Invalid voxel spacing: {voxel_spacing}"
        )

    return {
        "filename": path.name,
        "shape": tuple(int(x) for x in shape),
        "voxel_spacing": tuple(
            float(x) for x in voxel_spacing
        ),
        "affine": affine.tolist(),
        "orientation": nib.aff2axcodes(affine),
        "file_size_bytes": path.stat().st_size,
        "data_type": str(image.get_data_dtype()),
    }


def validate_matching_geometry(
    reference_path: str | Path,
    other_path: str | Path,
    affine_tolerance: float = 1e-3,
) -> Dict:
    """
    Verify that two NIfTI files occupy the same voxel grid.

    Intended for:
        MRI modality vs MRI modality
        MRI volume vs segmentation mask
    """

    reference = inspect_nifti(reference_path)
    other = inspect_nifti(other_path)

    if reference["shape"] != other["shape"]:
        raise NiftiValidationError(
            "Volume dimensions do not match. "
            f"Reference: {reference['shape']}, "
            f"Other: {other['shape']}."
        )

    reference_affine = np.asarray(
        reference["affine"], dtype=np.float64
    )

    other_affine = np.asarray(
        other["affine"], dtype=np.float64
    )

    if not np.allclose(
        reference_affine,
        other_affine,
        atol=affine_tolerance,
        rtol=0,
    ):
        raise NiftiValidationError(
            "MRI and segmentation mask have different "
            "spatial geometry (affine matrices)."
        )

    return {
        "compatible": True,
        "shape": reference["shape"],
        "voxel_spacing": reference["voxel_spacing"],
        "orientation": reference["orientation"],
    }


def validate_mri_study(
    modalities: Dict[str, str | Path],
) -> Dict:
    """
    Validate a four-modality BraTS-style MRI study.

    Required modalities:
        FLAIR, T1, T1ce, T2
    """

    required = {"FLAIR", "T1", "T1ce", "T2"}

    supplied = set(modalities.keys())

    missing = required - supplied
    unexpected = supplied - required

    if missing:
        raise NiftiValidationError(
            f"Missing MRI modalities: {sorted(missing)}"
        )

    if unexpected:
        raise NiftiValidationError(
            f"Unexpected modalities: {sorted(unexpected)}"
        )

    inspected = {
        name: inspect_nifti(path)
        for name, path in modalities.items()
    }

    reference_name = "FLAIR"
    reference_path = modalities[reference_name]

    for name, path in modalities.items():
        if name == reference_name:
            continue

        validate_matching_geometry(
            reference_path,
            path,
        )

    return {
        "valid": True,
        "reference_modality": reference_name,
        "modalities": inspected,
        "message": "All four MRI modalities have matching geometry.",
    }


def validate_segmentation_mask(
    reference_mri: str | Path,
    mask_path: str | Path,
) -> Dict:
    """
    Validate segmentation geometry and label values.

    Does not assume a particular BraTS label convention.
    """

    geometry = validate_matching_geometry(
        reference_mri,
        mask_path,
    )

    image = nib.load(str(mask_path))
    data = np.asanyarray(image.dataobj)

    if not np.isfinite(data).all():
        raise NiftiValidationError(
            "Segmentation mask contains non-finite values."
        )

    if not np.allclose(data, np.rint(data), atol=1e-5):
        raise NiftiValidationError(
            "Segmentation mask contains non-integer labels."
        )

    labels = np.unique(data)

    return {
        **geometry,
        "labels": [
            int(label) for label in labels
        ],
        "label_count": len(labels),
        "message": "Segmentation mask geometry and labels validated.",
    }
