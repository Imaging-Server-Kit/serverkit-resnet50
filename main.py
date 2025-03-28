"""
Algorithm server definition.
Documentation: https://github.com/Imaging-Server-Kit/cookiecutter-serverkit
"""

from typing import List, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit

import torch
from transformers import AutoImageProcessor, ResNetForImageClassification


class Parameters(BaseModel):
    """Defines the algorithm parameters"""

    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D, RGB).",
        json_schema_extra={"widget_type": "image"},
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array


class ResnetServer(serverkit.AlgorithmServer):
    def __init__(
        self,
        algorithm_name: str = "resnet50",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(self, image: np.ndarray, **kwargs) -> List[tuple]:
        """Runs the algorithm."""
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_label_idx]

        return [(predicted_label, {}, "class")]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Loads one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images


server = ResnetServer()
app = server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
