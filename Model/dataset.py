import json
from torch.utils.data import Dataset


class OcrDatasetFinetuning(Dataset):
    def __init__(self, data_list, tokenizer, empty_schema):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.empty_schema = empty_schema
        
    def __len__(self):
        return len(self.split_list)
    
    def __getitem__(self,index):
        input_text = create_word_bbox_text(self.data_list[index].word, self.data_list[index].bbox_quantized, self.empty_schema)
        outputs = json.dumps(self.data_list[index].entity.json())+'</Extraction>'
        encoding = self.tokenizer('\n'.join([input_text,outputs]), return_tensors="pt", truncation=True, padding=True,)
        return {i:v[0] for i,v in encoding.items()}
    

class OcrDatasetInference(Dataset):
    def __init__(self, data_list, tokenizer, empty_schema):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.empty_schema = empty_schema
        
    def __len__(self):
        return len(self.split_list)
    
    def __getitem__(self,index):
        input_text = create_word_bbox_text(self.data_list[index].word, self.data_list[index].bbox_quantized, self.empty_schema)
        encoding = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True,)
        return {i:v[0] for i,v in encoding.items()}


def create_string(
        txt: str,
        bbox: list[list[int]]
):
    return f"{txt} {bbox[0]}|{bbox[1]}"


def create_word_bbox_text(
        words: list[str],
        bboxes: list[list[int]],
        empty_schema: dict
):
    combined = [create_string(word, bbox) for bbox, word in zip(bboxes, words)]
    combined_rows = "\n".join(combined)
    return f"""
    <Document>
    {combined_rows}
    </Document>
    <Task>
    From the document, extract the text values and tags of the following entities:
    {json.dumps(empty_schema)}
    </Task>
    <Extraction>
    """