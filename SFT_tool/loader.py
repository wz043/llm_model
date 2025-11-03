import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

class Data_conversion(IterableDataset):
    def __init__(self, data, tokenizer, input_idex, history_idx, label_field, max_seqlen_padding, max_length_limit, add_special_token):
        self.data = data
        self.tokenizer = tokenizer
        self.input_idex = [input_idex] if type(input_idex) == str else input_idex
        self.history_idx = [history_idx] if type(history_idx) == str else history_idx
        self.label_field = [label_field] if type(label_field) == str else label_field
        self.max_seqlen = max_seqlen_padding
        self.max_length_limit = max_length_limit
        self.add_special_token = add_special_token

    def list_flatten(self,lst,tokenizer):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(self.list_flatten(item,tokenizer))
            else:
                result.append(item)
        return ["".join(result)]

    def data_conversion(self, tokenizer, input_idex, history_idx, label_field, max_seqlen, max_length_limit, add_special_token):
        for sample in self.data:
            parts = []
            for i in input_idex:
                val = sample.get(i, "")
                parts.append("".join(str(val) if val is not None else ""))
            parts = ["".join(parts)] if add_special_token == False else [f"{''.join(parts)}{tokenizer.eos_token}"]
            text_tensors = tokenizer(parts, padding=False, truncation=True, max_length=2048, return_tensors="pt",add_special_tokens=False)

            if history_idx is not None:
                for idx in history_idx:
                    history = self.list_flatten(sample[idx],tokenizer)
                    if text_tensors["input_ids"].shape[1] <= max_length_limit:
                        history_tensors = tokenizer(history, padding=False, truncation=True, max_length=2048, return_tensors="pt",add_special_tokens=False)["input_ids"]
                        cat_tensors = torch.cat([history_tensors[:,-(max_length_limit-text_tensors["input_ids"].shape[1]):],text_tensors["input_ids"]],dim=1)
                        text_tensors = {"input_ids":cat_tensors.to(text_tensors["input_ids"].dtype),"attention_mask":torch.ones_like(cat_tensors).to(text_tensors["attention_mask"].dtype)}

            text_input = text_tensors["input_ids"].squeeze(0)
            text_mask = text_tensors["attention_mask"].squeeze(0)

            if label_field == None:
                labels_tensors = text_tensors
                labels = labels_tensors["input_ids"].squeeze(0)
            else:
                label = []
                for i in label_field:
                    val = sample.get(i, "")
                    label.append("".join(str(val) if val is not None else ""))
                label = ["".join(label)] if add_special_token == False else [f"{''.join(label)}{tokenizer.eos_token}"]
                labels_tensors = tokenizer(label, padding=False, truncation=True, max_length=2048, return_tensors="pt",add_special_tokens=False)
                labels_tensors = labels_tensors["input_ids"].squeeze(0)
                mask = torch.zeros_like(text_input, dtype=torch.bool)
                for i in range(text_input.shape[0] - labels_tensors.shape[0] + 1):
                    if torch.all(text_input[i:i + labels_tensors.shape[0]] == labels_tensors):
                        mask[i:i + labels_tensors.shape[0]] = True
                labels = torch.where(mask, text_input, tokenizer.pad_token_id)


            if self.max_seqlen is not None:
                text_input = F.pad(text_input, [max_seqlen - text_input.shape[0], 0], mode='constant',
                                   value=tokenizer.pad_token_id) if text_input.shape[0] <= max_seqlen else text_input
                text_mask = F.pad(text_mask, [max_seqlen - text_mask.shape[0], 0], mode='constant', value=0) if \
                text_mask.shape[0] <= max_seqlen else text_mask
                labels = F.pad(labels, [max_seqlen - labels.shape[0], 0], mode='constant',value=tokenizer.pad_token_id) if labels.shape[0] <= max_seqlen else labels


            text_input = text_input[:max_length_limit]
            text_mask = text_mask[:max_length_limit]
            labels = labels[:max_length_limit]
            # print(input_ids,input_ids.shape,attn_mask,attn_mask.shape,labels,labels.shape)
            yield {"input_ids": text_input, "attention_mask": text_mask, "labels": labels}

    def __iter__(self):
        yield from self.data_conversion(self.tokenizer, self.input_idex, self.history_idx, self.label_field, self.max_seqlen,self.max_length_limit, self.add_special_token)


def Longgest_padding_collator(data, tokenizer):
    import copy
    data = copy.deepcopy(data)
    max_len = max(sample[idx].shape[0] for sample in data for idx in sample)
    for sample in data:
        for idx in sample:
            sample[idx] = F.pad(sample[idx], [max_len - len(sample[idx]), 0], mode='constant',value=0) if idx == 'attention_mask' else F.pad(sample[idx],[max_len - len(sample[idx]), 0],mode='constant',value=tokenizer.pad_token_id)
    batch_dict = {
        key: torch.stack([sample[key] for sample in data])
        for key in data[0].keys()
    }
    return batch_dict


