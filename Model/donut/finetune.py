import sys
sys.path = ["lmxd-python"]+sys.path

import os
from Model.donut.utils import load_model_and_tokenizer, get_special_tokens
from datasets.floorplan.preprocess import create_floorplan_document, load_all_results
from datasets.floorplan.schemas import RoomInfo, FloorplanEntity
from config import RESPONSE_FILEPATH, IMAGE_PATH, donut_training_args, DATA_FOLDER
from Model.donut.dataset import DonutFinetuning
from transformers import Seq2SeqTrainer
import logging

log = logging.getLogger('')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

log.info("Starting the process")
special_tokens = get_special_tokens([FloorplanEntity, RoomInfo])

log.info("Loading the model")
model, preprocessor = load_model_and_tokenizer("naver-clova-ix/donut-base" , special_tokens)

log.info("Loading the data")
all_entities = load_all_results(os.path.join(DATA_FOLDER, RESPONSE_FILEPATH))
documents = [
    create_floorplan_document(key=key,entity=all_entities[key]) for key in all_entities.keys()
]
finetuning_data = DonutFinetuning(documents[4:], preprocessor, os.path.join(DATA_FOLDER, IMAGE_PATH))


# hyperparameters used for multiple args
hf_repository_id = "donut-base-sroie"

log.info("Loading commenced")
# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=donut_training_args,
    train_dataset=finetuning_data,
)
trainer.train()

log.info("Saving the trained model")
trainer.save_model('trained_model')
