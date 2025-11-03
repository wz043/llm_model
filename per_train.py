import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_float32_matmul_precision('medium')
import transformers
from datasets import load_dataset
from model.train_model import Transformer, ModelArgs
import swanlab
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn.functional as F
from torch import utils,optim
swanlab.init(project="GMT_alpha_0.6B",
             workspace="jwz012",
             experiment_name="per_train_pro6000")

tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer/tokenizer", device_map="auto")
dataset = load_dataset("json", data_files="/root/autodl-fs/mobvoi_seq_monkey_general_open_corpus.jsonl", streaming=True)
train_data = dataset["train"]
print(next(iter(train_data))["text"])

from SFT_tool.loader import Data_conversion,Longgest_padding_collator
data_conversion = Data_conversion(data=train_data,
                                  tokenizer=tokenizer,
                                  input_idex=["text"],
                                  history_idx=None,
                                  label_field=None,
                                  max_seqlen_padding=None,
                                  max_length_limit=512,
                                  add_special_token=True)

train_loader = utils.data.DataLoader(data_conversion,
                                     batch_size=45,
                                     num_workers=0,
                                     collate_fn=lambda x: Longgest_padding_collator(x, tokenizer),
                                     pin_memory=True,
                                     drop_last=True,
                                     )

print(next(iter(train_loader)))
print(train_loader)

import lightning as L
from swanlab.integration.pytorch_lightning import SwanLabLogger
class Lit_structure(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.args = ModelArgs()
        self.model = Transformer(self.args)

    def _compute_accuracy(self, preds, labels):
        mask = labels != -100
        correct = (preds == labels) & mask
        return correct.sum().float() / mask.sum().float()

    def training_step(self, batch, labels=None, **kwargs):
        input_ids = batch["input_ids"]
        pad_mask = batch["attention_mask"].float()

        pad_mask = torch.einsum("bit,bjt->bij", pad_mask.unsqueeze(-1), pad_mask.unsqueeze(-1))
        pad_mask = torch.where(torch.isclose((1.0 - pad_mask.float()), torch.tensor(1.0)), -1e9, (1.0 - pad_mask))

        logits = self.model(input_ids,pad_mask)
        labels = batch["labels"]
        labels[labels == tokenizer.pad_token_id] = -100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        preds = shift_logits.argmax(dim=-1)
        acc = self._compute_accuracy(preds, shift_labels)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("epoch", float(self.current_epoch), on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("perplexity",torch.exp(loss).item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("accuracy_rate", acc, on_step=True, on_epoch=False, prog_bar=True)
        total_grad_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
            else:
                print(f"{name}: no_gradients")

        self.log('grad_norm/total', total_grad_norm, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            eps=1e-8
        )

        total_steps = self.trainer.max_steps
        warmup_steps = 200

        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps]
        )

        return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler,"interval": "step","frequency": 1}}

model = Lit_structure()

trainer = L.Trainer(devices="auto",
                    accelerator="gpu",
                    precision="bf16-mixed",
                    logger=SwanLabLogger(),
                    gradient_clip_val=5,
                    gradient_clip_algorithm="norm",
                    log_every_n_steps=1,
                    max_epochs=2,
                    accumulate_grad_batches=12,
                    max_steps=int((13000000//(train_loader.batch_size*12))*1),
                    overfit_batches=0,
                    callbacks=[
                        L.pytorch.callbacks.ModelCheckpoint(
                            dirpath="/root/autodl-fs/pertrain_checkpoints",
                            every_n_train_steps=1000,
                            save_top_k=3,
                            monitor="train_loss",
                            mode="min",
                            filename="model-{step}-{train_loss:.2f}",
                            save_last=True,
                        )
                    ]
                    )
trainer.fit(model=model, train_dataloaders=train_loader)

torch.save(model.state_dict(), "/root/autodl-fs/pertrain_checkpoints/model_weights.pth")