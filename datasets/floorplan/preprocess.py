import json

from pydantic import Field, BaseModel

from datasets.schemas.document import LxdmDocument, LxdmSplitDocument, Entity
from datasets.floorplan.ocr import OcrFileOutput


class RoomInfo(BaseModel):
    name: str| None = None
    dimension: str|None = None
    area: float| None = None


class FloorplanEntity(Entity):
    total_area: float| None = Field(None, alias="total area")
    rooms: list[RoomInfo]| None = None


class FloorplanDocument(LxdmDocument):
    entity: FloorplanEntity


class FloorplanSplitDocument(LxdmSplitDocument):
    entity: FloorplanEntity


def load_all_results(filepath):
    with open(filepath, "r") as f:
        label_results = json.load(f)

    return {k: FloorplanEntity.parse_obj(v) for k,v in label_results.items()}



def create_floorplan_document(
        document_ocr: OcrFileOutput | None,
        key: str,
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
