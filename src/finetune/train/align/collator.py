from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import DataCollatorForSeq2Seq

from transformers.data.data_collator import pad_without_fast_tokenizer_warning

@dataclass
class AlignDataCollatorWithPadding(DataCollatorForSeq2Seq):
    def _pad_labels(self, bs: int,  max_length: int, label_features: List[torch.LongTensor]) -> torch.Tensor:
        padded_labels = self.label_pad_token_id * torch.ones((bs, max_length), dtype=torch.long)
        for i in range(bs):
            padded_labels[i][max_length - label_features[i].size(-1): max_length] = label_features[i]
        return padded_labels.contiguous()  # in contiguous memory

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # concatenated_features = []

        input_features = []
        all_labels_lens = []
        label_features = []
        for feature in features:
            answer_len = len(feature["labels"]) - 1
            label_features.append(torch.LongTensor(feature["labels"][0:-1]))
            del feature["labels"]
            input_features.append(feature)
            all_labels_lens.append(answer_len)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        max_length = max(all_labels_lens)
        bs = len(all_labels_lens)
        batch["labels"] = self._pad_labels(bs, max_length, label_features)
        return batch
