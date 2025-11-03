import torch
import torch.nn.functional as F
from model.inference_model import GMT_Zero, ModelArgs
import transformers
import lightning as L
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GMT_1(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GMT_Zero(ModelArgs())

    def forward(self,x):
        return self.model(x)

model = GMT_1.load_from_checkpoint(checkpoint_path="./model_weight/post_train_weight.ckpt",strict=False).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer/tokenizer")

eos_token_id = torch.tensor([tokenizer.eos_token_id], device='cuda')
@torch.inference_mode()
def generate(tokenizer,input_ids,T,P,max_len:int=30):
    current_sequence = input_ids.clone()
    for i in range(max_len):
        logits = model(input_ids)[:,-1,:] / T
        probs = F.softmax(logits,dim=-1)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < P
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        sorted_probs[~nucleus] = 0
        sorted_probs /= sorted_probs.sum()
        select_ids = torch.multinomial(sorted_probs,num_samples=1)
        predict_ids = indices.gather(-1, select_ids)
        input_ids = torch.cat([current_sequence, predict_ids], dim=-1)
        current_sequence = input_ids

        if predict_ids == tokenizer.eos_token_id:
            break

    return tokenizer.decode(current_sequence[-1], skip_special_tokens=True)

while True:
    inference_text = input("enter your question:",)
    input_ids = tokenizer.encode(inference_text, return_tensors="pt",add_special_tokens=True).to('cuda')
    print(generate(tokenizer, input_ids, 0.6,0.95))