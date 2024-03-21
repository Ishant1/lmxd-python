import os.path

from tqdm import tqdm

from Model.dataset import OcrDatasetFinetuning
from Model.utils import load_model_and_tokenizer
from config import RESPONSE_FILEPATH, IMAGE_PATH, training_args
from datasets.floorplan.ocr import OcrEngine, load_ocr_from_file, save_ocr_result
from datasets.floorplan.preprocess import (
    create_floorplan_document, create_split_from_document, load_all_results, RoomInfo, FloorplanEntity
)
import transformers as tr

def run_finetune(
        entity_path: str = RESPONSE_FILEPATH,
        image_dir: str | None = IMAGE_PATH,
        ocr_result_path: str | None = None,
):

    if not (image_dir or ocr_result_path):
        raise ValueError("Either Images or OCR results need to be provided")

    all_entities = load_all_results(entity_path)
    all_keys = list(all_entities.keys())

    empty_dict = FloorplanEntity(total_area=0, rooms=[RoomInfo(name="", dimensions="", area=0)]).dict()

    ocr = OcrEngine()

    saved_ocr_results = load_ocr_from_file(ocr_result_path)
    split_documents = []
    print("creating ocr dataset")
    for key in tqdm(all_keys):
        if key not in saved_ocr_results:
            img_path = os.path.join(image_dir, key + '.jpeg')
            ocr_result = ocr.get_result_from_a_file(img_path)
            saved_ocr_results.update({key:ocr_result})
            if ocr_result_path:
                save_ocr_result(ocr_result_path, saved_ocr_results)

        floorplan_dataset = create_floorplan_document(
            saved_ocr_results[key], key, all_entities[key]
        )
        floorplan_split_dataset = create_split_from_document(floorplan_dataset)
        split_documents.append(floorplan_split_dataset)

    model, tokenizer = load_model_and_tokenizer()

    ocr_dataset = OcrDatasetFinetuning(
        data_list=split_documents,
        tokenizer=tokenizer,
        empty_schema=empty_dict
    )

    data_collator = tr.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = tr.Trainer(
        model,
        training_args,
        train_dataset=ocr_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(f"Training Model and saving at {training_args.output_dir}")
    trainer.train()

    return model


def run_inference(
    keys: list[str],
    image_dir: str = IMAGE_PATH,
    entity_path: str = RESPONSE_FILEPATH,
    ocr_result_path: str | None = None,
):
    pass
