from Model.donut.utils import load_model_and_tokenizer, get_special_tokens
from datasets.floorplan.preprocess import FloorplanEntity, RoomInfo, create_floorplan_document, load_all_results
from config import RESPONSE_FILEPATH, IMAGE_PATH
from datasets.floorplan.ocr import load_ocr_from_file


OCR_PATH = r"C:\dev\datasets\floorplan\ocr_results.json"

special_tokens = get_special_tokens([FloorplanEntity, RoomInfo])
model, preprocessor = load_model_and_tokenizer("naver-clova-ix/donut-base" , special_tokens)

all_entities = load_all_results(RESPONSE_FILEPATH)
saved_ocr_results = load_ocr_from_file(OCR_PATH)
documents = [
    create_floorplan_document(
            saved_ocr_results[key], key, all_entities[key]
        ) for key in all_entities.keys()
]





