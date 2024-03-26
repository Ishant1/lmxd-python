from Model.donut.utils import load_model_and_tokenizer, get_special_tokens
from datasets.floorplan.preprocess import FloorplanEntity, RoomInfo, create_floorplan_document, load_all_results
from config import RESPONSE_FILEPATH, IMAGE_PATH
from Model.donut.dataset import DonutFinetuning
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from logging import getLogger

logger = getLogger(name="finetuning_logger")

logger.info("Starting the process")
special_tokens = get_special_tokens([FloorplanEntity, RoomInfo])

logger.info("Loading the model")
model, preprocessor = load_model_and_tokenizer("naver-clova-ix/donut-base" , special_tokens)

logger.info("Loading the data")
all_entities = load_all_results(RESPONSE_FILEPATH)
documents = [
    create_floorplan_document(key=key,entity=all_entities[key]) for key in all_entities.keys()
]
finetuning_data = DonutFinetuning(documents, preprocessor, IMAGE_PATH)


# hyperparameters used for multiple args
hf_repository_id = "donut-base-sroie"

# Arguments for training
training_args = Seq2SeqTrainingArguments(
    output_dir="donut-training",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    fp16=True,
    logging_steps=20,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    # push to hub parameters
    report_to=None,
)

logger.info("Loading commenced")
# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=finetuning_data,
)

logger.info("Saving the trained model")
trainer.model.save_pretrained('trained_model')
