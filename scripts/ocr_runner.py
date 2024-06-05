import json
import os.path
import random
import shutil

from tqdm import tqdm

from datasets.floorplan.ocr import OcrEngine
from datasets.floorplan.preprocess import (
    load_all_results, match_labels_and_linking
)

DATA_DIR = r"C:\dev\datasets\floorplan\GeoLayDir\val"
IMAGE_PATH = r"C:\dev\datasets\floorplan\images"
LABEL_PATH = rf"C:\dev\datasets\floorplan\accepted_responses.json"
split = 1


def setup_dir(data_dir=DATA_DIR):
    training_dir = os.path.join(data_dir, 'training_data')
    testing_dir = os.path.join(data_dir, 'testing_data')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(os.path.join(training_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(training_dir, 'annotations'), exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)
    os.makedirs(os.path.join(testing_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(testing_dir, 'annotations'), exist_ok=True)
    return training_dir, testing_dir


def get_image_paths_and_labels(image_dir=IMAGE_PATH, label_path=LABEL_PATH):
    sample = 50
    image_paths = {i.split('.')[0]: fr"{IMAGE_PATH}\{i}" for i in os.listdir(image_dir)}
    labelled_data = load_all_results(label_path, unique=False)

    unlabelled_images = {
        i: {"image":v, "annotation":labelled_data.get(i,None)}
        for i, v in image_paths.items() if i not in labelled_data
    }
    random.seed(42)
    sample_keys = random.sample(unlabelled_images.keys(), sample)

    return dict(filter(lambda a: a[0] in sample_keys, unlabelled_images.items()))


def main():
    training_dir, testing_dir = setup_dir()
    imagepath_dict = get_image_paths_and_labels()
    ocr = OcrEngine()
    test_sample_keys = random.sample(imagepath_dict.keys(), int(len(imagepath_dict)*split))

    for i in tqdm(imagepath_dict):
        ocr_df = ocr.get_result_from_a_file(imagepath_dict.get(i).get("image"), block=True)
        ocr_labels = match_labels_and_linking(ocr_df, imagepath_dict.get(i).get("annotation"))

        # print(ocr_labels)

        base_dir = testing_dir if i in test_sample_keys else training_dir
        ann_file = os.path.join(base_dir, 'annotations', f'{i}.json')
        img_dir = os.path.join(base_dir, 'images')
        with open(ann_file, 'w+') as f:
            json.dump(ocr_labels, f)
        shutil.copyfile(rf'{IMAGE_PATH}\{i}.jpeg', rf'{img_dir}\{i}.jpeg')


if __name__ == "__main__":
    main()





