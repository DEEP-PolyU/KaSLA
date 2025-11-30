import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional

from ..extras.misc import use_modelscope


if TYPE_CHECKING:
    from ..hparams import DataArguments


import hashlib

def get_sha1(file_path):
    sha1_hash = hashlib.sha1()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha1_hash.update(chunk)
    return sha1_hash.hexdigest()

@dataclass
class DatasetAttr:
    load_from: Literal["hf_hub", "ms_hub", "script", "file"]
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: Optional[bool] = False
    formatting: Optional[Literal["alpaca", "sharegpt"]] = "alpaca"

    system: Optional[str] = None

    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    messages: Optional[str] = "conversations"
    tools: Optional[str] = None

    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"

    def __repr__(self) -> str:
        return self.dataset_name


def get_dataset_list(data_args: "DataArguments") -> List["DatasetAttr"]:
    dataset_names = [ds.strip() for ds in data_args.dataset.split(",")] if data_args.dataset is not None else []

    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [float(prob.strip()) for prob in data_args.interleave_probs.split(",")]

    dataset_list: List[DatasetAttr] = []
    for dataset_name in dataset_names:

        dataset_attr = DatasetAttr(
            "file",
            dataset_name=dataset_name,
            dataset_sha1=get_sha1(os.path.join(data_args.dataset_dir, dataset_name)),
        )
        dataset_attr.subset =None
        dataset_attr.folder = None
        dataset_attr.ranking = False
        dataset_attr.formatting = "alpaca"

        dataset_list.append(dataset_attr)

    return dataset_list
