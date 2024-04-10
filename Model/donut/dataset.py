from torch.utils.data import Dataset
import operator
from PIL import Image
import numpy as np
import cv2


class DonutFinetuning(Dataset):
    def __init__(self, data_list, processor, image_dir):
        self.data_list = data_list
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ignore_id = -100
        max_length = 512
        data_point = self.data_list[index]
        entity_dict = data_point.entity.dict()
        entity_dict['rooms'] = sorted(entity_dict['rooms'], key=operator.itemgetter('name'))

        image = load_image_from_local(
            key=data_point.key,
            image_dir=self.image_dir
        )

        target_text = json2token(
            entity=entity_dict
        )

        try:
            pixel_values = self.processor(
                image, return_tensors="pt"
            ).pixel_values.squeeze()
        except Exception as e:
            print(data_point)
            print(f"Error: {e}")
            return {}

        # tokenize document
        input_ids = self.processor.tokenizer(
            target_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
        return {"pixel_values": pixel_values, "labels": labels, "target_sequence": target_text}


def json2token(
        entity: dict
):
    obj = entity
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            for k in obj.keys():
                output += (
                        fr"<s_{k}>"
                        + json2token(obj[k])
                        + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        return obj


def load_image_from_local(key: str, image_dir: str):
    img = Image.open(f"{image_dir}/{key}.jpeg")
    numpy_img = np.array(img)
    if len(numpy_img.shape) == 2:
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_GRAY2RGB)
    elif len(numpy_img.shape) == 3 and numpy_img.shape[2] != 3:
        numpy_img = numpy_img.mean(axis=2).astype("uint8")
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_GRAY2RGB)
    return numpy_img
