import json

from pydantic import Field, BaseModel
import re

from datasets.schemas.document import LxdmDocument, LxdmSplitDocument, Entity
from datasets.floorplan.ocr import OcrFileOutput
import numpy as np


class RoomInfo(BaseModel):
    name: str| None = None
    dimension: list[str]|None = None


class FloorplanEntity(Entity):
    total_area: float| None = Field(None, alias="total area")
    rooms: list[RoomInfo]| None = None


class FloorplanDocument(LxdmDocument):
    entity: FloorplanEntity


class FloorplanSplitDocument(LxdmSplitDocument):
    entity: FloorplanEntity


def name_validator(name):
    clean_name = re.sub("\d","",name).strip().lower()
    if re.findall("bed|suite", clean_name):
        return "bedroom"
    elif re.findall("reception|lounge|living", clean_name):
        if re.findall("kitchen|diner", clean_name):
            return "open plan living room"
        else:
            return "living room"
    elif re.findall("kitchen|diner|dining", clean_name):
        return "kitchen"

    else:
        return clean_name


def load_all_results(filepath):
    with open(filepath, "r") as f:
        label_results = json.load(f)

    cleaned_labels = {}
    for key, label_values in label_results.items():
        rooms = label_values['rooms']
        cleaned_rooms = []
        for r in rooms:
            r['name'] = name_validator(r['name'])
            cleaned_rooms.append(r)

        unqiue_rooms = np.unique([x["name"] for x in cleaned_rooms])
        unique_rooms_dict = []
        for room_section in unqiue_rooms:
            filtered_rooms = list(filter(lambda x: x['name'] == room_section, cleaned_rooms))
            dimension_list = [x['dimension'] for x in filtered_rooms if x['dimension']]
            unique_rooms_dict.append({'name': room_section, 'dimension': dimension_list})
        label_values['rooms'] = unique_rooms_dict
        cleaned_labels[key] = FloorplanEntity.parse_obj(label_values)

    return cleaned_labels



def create_floorplan_document(
        key: str,
        document_ocr: OcrFileOutput | None = None,
        entity: FloorplanEntity | None = None,
) -> FloorplanDocument:
    floorplan_document = FloorplanDocument(
        key=key,
        bbox= [ocr_output.bbox for ocr_output in document_ocr.ocr_result] if document_ocr else None,
        word= [ocr_output.text for ocr_output in document_ocr.ocr_result] if document_ocr else None,
        entity=entity
    )

    return floorplan_document


def create_split_from_document(
        document: FloorplanDocument
) -> FloorplanSplitDocument:

    floorplan_document_dict = document.dict()
    floorplan_document_dict.update({'index': 0})
    floorplan_split = FloorplanSplitDocument.parse_obj(
        floorplan_document_dict
    )
    return floorplan_split
