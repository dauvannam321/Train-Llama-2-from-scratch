import torch
from Llama2 import *

torch.manual_seed(0)

allow_cuda = False
device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

model = LLaMA.build(
    checkpoints_dir='llama-2-7b',
    tokenizer_path='tokenizer.model',
    load_model=False,
    max_seq_len=256,
    max_batch_size=3,
    device=device
)

prompts = [
    # "What is Machine Learning ?",
    # "1 + 1 = ?",
    # "Once upon a time, there was a little girl named Lola.",
    "Once upon a time, in a big forest",
]

# tokenizer = SentencePieceProcessor()
# tokenizer.load('tokenizer.model')
# for i in [1, 9038, 2501, 263, 931, 29892, 297, 263, 4802, 13569]:
#     print(i, tokenizer.Decode(i))
# res = tokenizer.encode("Once upon a time, there was a little girl named Lola.", out_type=int, add_bos=True, add_eos=True)
# print(res)

# model.summary()

out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
# # assert len(out_texts) == len(prompts)
# # print(out_texts)
for i in range(len(out_texts)):
    print(f'{out_texts[i]}')
    print('-' * 100)
print("ALL OK!")