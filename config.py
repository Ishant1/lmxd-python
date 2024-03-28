import tempfile
import transformers as tr

tmpdir = tempfile.TemporaryDirectory()
local_training_root = tmpdir.name

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