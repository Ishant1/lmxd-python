import tempfile
import transformers as tr

tmpdir = tempfile.TemporaryDirectory()
local_training_root = tmpdir.name

DATA_FOLDER = r"drive\MyDrive\dataset\floorplan"
IMAGE_PATH = r"accepted_images"
RESPONSE_FILEPATH = r"accepted_responses.json"
MODEL_PATH = r"C:\dev\models\gpt2"

training_args = tr.TrainingArguments(
        local_training_root+'\checkpoints',
        num_train_epochs=3,  # default number of epochs to train is 3
        per_device_train_batch_size=4,
        logging_steps=25,
        eval_steps=50,
        optim="adamw_torch",
        report_to=None,
    )

# Arguments for training
donut_training_args = tr.Seq2SeqTrainingArguments(
    output_dir="donut-training",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    weight_decay=0.01,
    fp16=True,
    logging_steps=20,
    save_total_limit=1,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    # push to hub parameters
    report_to=None,
)