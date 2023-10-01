import torch
import transformers
from transformer_utils.low_memory import enable_low_memory_load

enable_low_memory_load()

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl')

def text_to_input_ids(text):
    toks = tokenizer.encode(text)
    return torch.as_tensor(toks).view(1, -1).cuda()

input_ids = text_to_input_ids("This is an example. You can probably think of a more fun text to use than this one.")

plot_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=45)  # logits

plot_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=45, probs=True)  # probabilities

plot_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=45, kl=True)  # K-L divergence