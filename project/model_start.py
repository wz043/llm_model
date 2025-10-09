import torch
from model.inference_model import GMT_Zero, ModelArgs
import transformers
import torch.nn.functional as F
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
model = GMT_Zero(ModelArgs())
print("test")
quant_weights = torch.load("./model_weights.pth", map_location='cuda')
model.load_state_dict(quant_weights, strict=False)
model.eval()
print("loading_success")

tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer/tokenizer")
eos_token_id = torch.tensor([tokenizer.eos_token_id], device='cuda')

@torch.no_grad()
def generate(tokenizer, input_ids, max_length: int = 100):
    current_sequence = input_ids.clone()
    total_step = 0
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)[:, -1, :]
            top_values, _ = torch.topk(logits, k=10)
            logits = logits.clone()
            logits[logits < top_values[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            current_sequence = torch.cat([current_sequence, idx_next], dim=-1)
            total_step +=1
            print("running_step",total_step)

        if idx_next.item() == eos_token_id:
            break

    return tokenizer.decode(current_sequence[0], skip_special_tokens=True)


while True:
    inference_text = input("enter your question:",)
    input_ids = tokenizer.encode(inference_text, return_tensors="pt",add_special_tokens=True).to('cuda')
    print(input_ids)
    print(generate(tokenizer, input_ids))


