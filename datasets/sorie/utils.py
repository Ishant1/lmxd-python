from datasets import load_dataset


def load_sorie_dataset(ratio=0.5, save_loc=None):
    dataset = load_dataset("darentang/sroie", split=f'train[:{ratio*100}%]', features={'id':'string','words':'sequence','bboxes':'sequence','ner_tags':'sequence'})
    if save_loc:
        dataset.save_to_disk(save_loc)
    return dataset.to_dict()


TOTAL      0.268519
DATE       0.638690
ADDRESS    0.823241
COMPANY    0.787204