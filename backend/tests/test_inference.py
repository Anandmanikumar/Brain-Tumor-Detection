
import tempfile
from pathlib import Path

from app.services.inference import (
    create_inference_engine,
    InferenceError,
)


def main():

    with tempfile.TemporaryDirectory() as temp:

        root = Path(temp)

        input_dir = root / "input"
        output_dir = root / "output"

        input_dir.mkdir()
        output_dir.mkdir()

        # Empty placeholder files are sufficient for this
        # mock-engine workflow test. They are NOT valid MRI.
        for modality in ["FLAIR", "T1", "T1ce", "T2"]:
            (input_dir / f"{modality}.nii.gz").touch()

        engine = create_inference_engine("mock")

        result = engine.predict(
            input_dir=input_dir,
            output_dir=output_dir,
            case_id="TEST_00001",
        )

        print("Mock inference completed.")
        print("Artifact:", result)
        print(result.read_text(encoding="utf-8"))

        assert result.exists()
        assert result.name == "MOCK_INFERENCE.txt"

        try:
            create_inference_engine("nnunet")

        except ValueError:
            print(
                "Real nnU-Net correctly requires "
                "a verified Docker command."
            )

        print("\nALL INFERENCE TESTS PASSED")


if __name__ == "__main__":
    main()
