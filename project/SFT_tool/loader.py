import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

class Data_conversion(IterableDataset):
    def __init__(self,data,tokenizer,input_idex,label_field,max_seqlen_padding,max_length_limit):
        self.data = data
        self.tokenizer = tokenizer
        self.input_idex = input_idex
        self.label_field = label_field
        self.max_seqlen = max_seqlen_padding
        self.max_length_limit = max_length_limit

    def data_conversion(self,tokenizer,input_idex,label_field,max_seqlen,max_length_limit):
        for sample in self.data:
            parts = []
            for i in input_idex:
                val = sample.get(i, "")
                parts.append("".join(str(val) if val is not None else ""))
            parts = ["".join(parts)]

            text_tensors = tokenizer(parts, padding=False, truncation=True, max_length=2048, return_tensors="pt", add_special_tokens=True)
            labels_tensors = tokenizer(str(sample[label_field]), padding=False, truncation=True, max_length=2048, return_tensors="pt", add_special_tokens=False)

            text_input = text_tensors["input_ids"].squeeze(0)
            text_mask = text_tensors["attention_mask"].squeeze(0)
            labels_tensors = labels_tensors["input_ids"].squeeze(0)

            mask = torch.zeros_like(text_input, dtype=torch.bool)
            for i in range(text_input.shape[0] - labels_tensors.shape[0] + 1):
                if torch.all(text_input[i:i + labels_tensors.shape[0]] == labels_tensors):
                    mask[i:i + labels_tensors.shape[0]] = True

            labels = torch.where(mask, text_input, tokenizer.pad_token_id)
            if self.max_seqlen is not None:
                text_input = F.pad(text_input,[max_seqlen - text_input.shape[0],0],mode='constant',value=tokenizer.pad_token_id) if text_input.shape[0] <= max_seqlen else text_input[:max_seqlen]
                text_mask = F.pad(text_mask,[max_seqlen - text_mask.shape[0],0],mode='constant',value=0) if text_mask.shape[0] <= max_seqlen else text_mask[:max_seqlen]
                labels = F.pad(labels,[max_seqlen-labels.shape[0],0],mode='constant',value=tokenizer.pad_token_id) if labels.shape[0] <= max_seqlen else labels[:max_seqlen]
            #print(input_ids,input_ids.shape,attn_mask,attn_mask.shape,labels,labels.shape)
            if self.max_length_limit is not None:
                text_input = text_input[:max_length_limit]
                text_mask = text_mask[:max_length_limit]
                labels = labels[:max_length_limit]
            yield {"input_ids":text_input,"attention_mask":text_mask,"labels":labels}

    def __iter__(self):
        yield from self.data_conversion(self.tokenizer,self.input_idex,self.label_field,self.max_seqlen,self.max_length_limit)

def Longgest_padding_collator(data,tokenizer):
    import copy
    data = copy.deepcopy(data)
    max_len = max(sample[idx].shape[0] for sample in data for idx in sample)
    for sample in data:
        for idx in sample:
            sample[idx] = F.pad(sample[idx],[max_len - len(sample[idx]),0],mode='constant',value=0) if idx == 'attention_mask' else F.pad(sample[idx],[max_len - len(sample[idx]),0],mode='constant',value=tokenizer.pad_token_id)
    batch_dict = {
                  key: torch.stack([sample[key] for sample in data])
                  for key in data[0].keys()
    }
    return batch_dict



