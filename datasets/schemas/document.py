from pydantic import model_validator, BaseModel
import numpy as np

from datasets.utils import get_bins_from_list


class Entity(BaseModel):
    ...


class LxdmDocument(BaseModel):
    key: str
    bbox: list | None = None
    word: list | None = None
    entity: Entity | None = None
    bbox_quantized: list | None = None

    @model_validator(mode="before")
    def compute_bbox_quantized(cls, values):
        if values['bbox']:
            bbox_means = [[np.mean([b[0], b[2]]), np.mean([b[1], b[3]])] for b in
                          values.get('bbox')]
            values['bbox_quantized'] = np.apply_along_axis(get_bins_from_list, arr=bbox_means, axis=0).tolist()
        return values


class LxdmSplitDocument(BaseModel):
    key: str
    index: int
    bbox: list | None = None
    word: list | None = None
    bbox_quantized: list | None = None
    entity: Entity | None = None
