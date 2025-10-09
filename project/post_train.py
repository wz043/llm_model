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
             experiment_name="post_train_a_1")

tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer/tokenizer", device_map="auto")
dataset = load_dataset("json", data_files=["./alpaca_gpt4_data_zh.json","./belle_data1M_cn.json","./CoT_Chinese_data.json","./HC3_Chinese_ChatGPT.json"], streaming=True)
train_data = dataset["train"]

from SFT_tool.loader import Data_conversion,Longgest_padding_collator
data_conversion = Data_conversion(train_data,tokenizer,["instruction","input","output"],"output",None,256)

train_loader = utils.data.DataLoader(data_conversion,
                                     batch_size=15,
                                     num_workers=0,
                                     collate_fn=lambda x: Longgest_padding_collator(x, tokenizer),
                                     pin_memory=True,
                                     drop_last=True,
                                     )

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

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=5e-5,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        total_steps = self.trainer.max_steps
        warmup_steps = 500

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


model = Lit_structure.load_from_checkpoint(checkpoint_path="/root/autodl-fs/model_weight/model-step=1995000-train_loss=1.93.ckpt",strict=False)

trainer = L.Trainer(devices="auto",
                    accelerator="gpu",
                    precision="bf16-mixed",
                    gradient_clip_val=10.0,
                    gradient_clip_algorithm="norm",
                    logger=SwanLabLogger(),
                    log_every_n_steps=1,
                    max_epochs=3,
                    min_epochs=2,
                    max_steps=int((1141111//train_loader.batch_size)*1.5),
                    overfit_batches=0,
                    callbacks=[
                        L.pytorch.callbacks.ModelCheckpoint(
                            dirpath="/root/autodl-fs/post_train_weight",
                            every_n_train_steps=5000,
                            save_top_k=3,
                            monitor="train_loss",
                            filename="model-{step}-{train_loss:.2f}",
                            save_last=True,
                            save_on_train_epoch_end=True
                        )
                    ]
                    )

trainer.fit(model=model, train_dataloaders=train_loader)

torch.save(model.state_dict(), "/root/autodl-fs/post_train_weight/last_model_weights.pth")