from pydantic import model_validator, BaseModel
import numpy as np

from datasets.utils import get_bins_from_list


class Entity(BaseModel):
    ...


class LxdmDocument(BaseModel):
    key: str
    bbox: list
    word: list
    entity: Entity | None
    bbox_quantized: list

    @model_validator(mode="before")
    def compute_bbox_quantized(cls, values):
        bbox_means = [[np.mean([b[0], b[2]]), np.mean([b[1], b[3]])] for b in
                      values.get('bbox')]
        values['bbox_quantized'] = np.apply_along_axis(get_bins_from_list, arr=bbox_means, axis=0).tolist()
        return values


class LxdmSplitDocument(BaseModel):
    key: str
    index: int
    bbox: list
    word: list
    bbox_quantized: list
    entity: Entity | None
