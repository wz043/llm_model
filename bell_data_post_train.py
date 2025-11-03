import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_float32_matmul_precision('medium')
import transformers
from datasets import load_dataset
from model.train_model import Transformer, ModelArgs
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn.functional as F
from torch import utils,optim
import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger
swanlab.init(project="GMT_alpha_0.6B",
             workspace="jwz012",
             experiment_name="post_train_b_2")

tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer/tokenizer", device_map="auto")
dataset = load_dataset("json", data_files="/root/autodl-fs/belle_data1M_cn.json", streaming=True)
train_data = dataset["train"]
print(next(iter(train_data))["input"])

from SFT_tool.loader import Data_conversion,Longgest_padding_collator
data_conversion = Data_conversion(data=train_data,
                                  tokenizer=tokenizer,
                                  input_idex=["instruction","input","output"],
                                  history_idx=None,
                                  label_field=["output"],
                                  max_seqlen_padding=None,
                                  max_length_limit=256,
                                  add_special_token=True)

train_loader = utils.data.DataLoader(data_conversion,
                                     batch_size=90,
                                     num_workers=0,
                                     collate_fn=lambda x: Longgest_padding_collator(x, tokenizer),
                                     pin_memory=True,
                                     drop_last=True,
                                     )

val_dataset = load_dataset("json", data_files="/root/autodl-fs/sft_data_zh.jsonl", streaming=True)
val_data = val_dataset["train"]
val_conversion = Data_conversion(data=val_data,
                                  tokenizer=tokenizer,
                                  input_idex=["instruction","input","output"],
                                  history_idx=None,
                                  label_field=["output"],
                                  max_seqlen_padding=None,
                                  max_length_limit=256,
                                  add_special_token=True)

val_loader = utils.data.DataLoader(val_conversion,
                                   batch_size=90,
                                   num_workers=0,
                                   collate_fn=lambda x: Longgest_padding_collator(x, tokenizer),
                                   pin_memory=True,
                                   drop_last=True,
                                   )

print(next(iter(train_loader)))

import lightning as L
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

        if torch.isnan(loss).any():
            print("NaN happend")

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

    def validation_step(self, batch, labels=None, **kwargs):
        input_ids = batch["input_ids"]
        pad_mask = batch["attention_mask"].float()

        pad_mask = torch.einsum("bit,bjt->bij", pad_mask.unsqueeze(-1), pad_mask.unsqueeze(-1))
        pad_mask = torch.where(torch.isclose((1.0 - pad_mask.float()), torch.tensor(1.0)), -1e9, (1.0 - pad_mask))

        logits = self.model(input_ids, pad_mask)
        labels = batch["labels"]
        labels[labels == tokenizer.pad_token_id] = -100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        predicted_ids = shift_logits.argmax(dim=-1)

        print("input================================================================================")
        print(tokenizer.decode(input_ids[0]))
        print("predict==============================================================================")
        print(tokenizer.decode(predicted_ids[0]))

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_perplexity", torch.exp(loss).item(), on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=2e-4,
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
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps]
        )

        return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler,"interval": "step","frequency": 1}}


model = Lit_structure.load_from_checkpoint(checkpoint_path="/root/autodl-fs/pertrain_checkpoints/model-step=14000-train_loss=2.79.ckpt",strict=False)

trainer = L.Trainer(devices="auto",
                    accelerator="gpu",
                    precision="bf16-mixed",
                    logger=SwanLabLogger(),
                    gradient_clip_val=5,
                    gradient_clip_algorithm="norm",
                    log_every_n_steps=1,
                    max_epochs=100,
                    accumulate_grad_batches=6,
                    max_steps=int((1000000//(train_loader.batch_size*6))*50),
                    val_check_interval=1000,
                    limit_val_batches=10,
                    overfit_batches=0,
                    callbacks=[
                        L.pytorch.callbacks.ModelCheckpoint(
                            dirpath="/root/autodl-fs/post_train_checkpoints",
                            every_n_train_steps=1000,
                            save_top_k=3,
                            monitor="train_loss",
                            mode="min",
                            filename="model-{step}-{train_loss:.2f}",
                            save_last=True,
                        )
                    ]
                    )

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

torch.save(model.state_dict(), "/root/autodl-fs/post_train_checkpoints/last_weight.pth")
swanlab.finish()