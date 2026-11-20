
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class InferenceError(RuntimeError):
    """Raised when model inference fails."""
    pass


class InferenceEngine(ABC):
    """Common interface for all inference engines."""

    @abstractmethod
    def predict(
        self,
        input_dir: Path,
        output_dir: Path,
        case_id: str,
    ) -> Path:
        pass


class MockInferenceEngine(InferenceEngine):
    """
    Testing engine.

    It deliberately does NOT generate a tumor mask.
    It creates a status artifact so the application workflow
    can be tested without claiming segmentation occurred.
    """

    def predict(
        self,
        input_dir: Path,
        output_dir: Path,
        case_id: str,
    ) -> Path:

        output_dir.mkdir(parents=True, exist_ok=True)

        required = [
            "FLAIR",
            "T1",
            "T1ce",
            "T2",
        ]

        for modality in required:
            files = (
                list(input_dir.glob(f"{modality}.nii"))
                + list(input_dir.glob(f"{modality}.nii.gz"))
            )

            if len(files) != 1:
                raise InferenceError(
                    f"Expected exactly one {modality} input."
                )

        artifact = output_dir / "MOCK_INFERENCE.txt"

        artifact.write_text(
            "MOCK INFERENCE COMPLETED\n"
            f"Case ID: {case_id}\n"
            "No tumor segmentation was performed.\n"
            "This artifact is not a medical result.\n",
            encoding="utf-8",
        )

        return artifact


class NNUNetDockerEngine(InferenceEngine):
    """
    Docker-based nnU-Net inference engine.

    Configure NNUNET_COMMAND as a JSON array in the
    backend environment. Use {input_dir}, {output_dir}
    and {case_id} as placeholders.

    This implementation is not enabled until the exact
    working Docker command and model configuration
    have been verified on the IIT server.
    """

    def __init__(
        self,
        command_template: list[str],
        timeout_seconds: int = 3600,
    ):
        if not command_template:
            raise ValueError("Docker command cannot be empty.")

        self.command_template = command_template
        self.timeout_seconds = timeout_seconds

    def predict(
        self,
        input_dir: Path,
        output_dir: Path,
        case_id: str,
    ) -> Path:

        input_dir = input_dir.resolve()
        output_dir = output_dir.resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        replacements = {
            "{input_dir}": str(input_dir),
            "{output_dir}": str(output_dir),
            "{case_id}": case_id,
        }

        command = []

        for argument in self.command_template:
            for placeholder, value in replacements.items():
                argument = argument.replace(
                    placeholder,
                    value,
                )

            command.append(argument)

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )

        except subprocess.TimeoutExpired as exc:
            raise InferenceError(
                "nnU-Net inference timed out."
            ) from exc

        except OSError as exc:
            raise InferenceError(
                f"Unable to start Docker inference: {exc}"
            ) from exc

        if completed.returncode != 0:
            raise InferenceError(
                "nnU-Net inference failed.\n"
                f"Exit code: {completed.returncode}\n"
                f"Error: {completed.stderr[-4000:]}"
            )

        results = sorted(
            list(output_dir.glob("*.nii.gz"))
            + list(output_dir.glob("*.nii"))
        )

        if len(results) != 1:
            raise InferenceError(
                "Expected exactly one segmentation output, "
                f"found {len(results)}."
            )

        return results[0]


def create_inference_engine(
    mode: str = "mock",
    command_template: list[str] | None = None,
) -> InferenceEngine:

    if mode == "mock":
        return MockInferenceEngine()

    if mode == "nnunet":
        if not command_template:
            raise ValueError(
                "A verified nnU-Net Docker command is required."
            )

        return NNUNetDockerEngine(
            command_template=command_template
        )

    raise ValueError(
        f"Unsupported inference mode: {mode}"
    )
