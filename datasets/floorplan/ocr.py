import json
import os.path

import cv2
from pydantic import BaseModel, field_validator
from PIL import Image
from skimage import io
import numpy as np
from paddleocr import PaddleOCR

from datasets.utils import combine_bbox

OCR_ENGINE = r"C:\Users\iagg1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


class OcrTextOutout(BaseModel):
    bbox: list[float]
    text: str
    confidence: float

    @field_validator("bbox", mode="before")
    def convert_bbox_into_four(cls, v):
        if isinstance(v[0], list):
            v = combine_bbox(v)
        return v


class OcrFileOutput(BaseModel):
    filename: str
    ocr_result: list[OcrTextOutout]


class OcrEngine:
    def __init__(self, local_image=True) -> None:
        self.engine = None
        self.terradata_file:str = None

        if local_image:
            self.loader = self._load_local_image
        else:
            self.loader = self._load_web_image

        self.setup_ocr()

    def setup_ocr(self) -> None:
        self.engine = PaddleOCR(use_angle_cls=True, lang='en')
        # pytesseract.pytesseract.tesseract_cmd = OCR_ENGINE

    @staticmethod
    def preprocess_image(img: np.ndarray) -> np.ndarray:
        if len(img.shape)==3 and img.shape[2]==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _load_local_image(img_path: str) -> np.ndarray:
        return np.array(Image.open(img_path))

    @staticmethod
    def _load_web_image(image_path: str) -> np.ndarray:
        return io.imread(image_path)

    def process_image(self, img: np.ndarray) -> str:
        bbox_and_text = self.engine.ocr(img)
        all_outputs = []
        for ocr_output in bbox_and_text[0]:
            ocr_text_output = OcrTextOutout(
                bbox = ocr_output[0],
                text = ocr_output[1][0],
                confidence = ocr_output[1][1]
            )
            all_outputs.append(ocr_text_output)
        return all_outputs

    def get_result_from_a_file(self, image_path: str):
        # img = self.loader(image_path)
        # processed_img = self.preprocess_image(img)
        ocr_output = self.process_image(image_path)
        ocr_result = OcrFileOutput(
            filename = image_path,
            ocr_result = ocr_output
        )
        return ocr_result


def load_ocr_from_file(filename: str | None) -> dict[str,OcrFileOutput]:
    ocr_results = {}
    if filename and os.path.exists(filename):
        with open(filename, "r") as f:
            ocr_results = json.load(f)

    return {i:OcrFileOutput.parse_obj(v) for i,v in ocr_results.items()}


def save_ocr_result(
        filename: str,
        ocr_results: dict[str,OcrFileOutput]
) -> None:
    ocr_result_nested_dict = {i:v.dict() for i,v in ocr_results.items()}

    with open(filename, "w+") as f:
        json.dump(ocr_result_nested_dict, f)
