from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from modules import ModelArgs, Transformer


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, 
              max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints files found"

            chk_path = checkpoints[0]
            print(f'Loading checkpoint {chk_path}')
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f'Loaded checkpoint in {(time.time() - prev_time):.2f}s')
            prev_time = time.time()
        
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f}s")

        return LLaMA(model, tokenizer, model_args)
    
    def summary(self):
        total_params = 0
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} --- Parameters: {param.numel()}")
            total_params += param.numel()
        
        print("Total parameters:", total_params)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        # print("*" * 20 + "Inference" + "*" * 20)
        # print(prompts)

        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        print("prompt_tokens: ",prompt_tokens)
        
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        print("Max prompt len: ", max_prompt_len)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        print("Total len: ", total_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        print("Pad id: ", pad_id)
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        print("Begin Tokens: ", tokens)

        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        print("tokens[k, : len(t)]: ", tokens)
        # print(tokens.shape)
        # return
        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        print("eos_reached: ", eos_reached)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        print("prompt_tokens_mask: ", prompt_tokens_mask)
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            print("\nCur_pos: ", cur_pos)
            with torch.no_grad():
                print("tokens[:, cur_pos-1:cur_pos]: ", tokens[:, cur_pos-1:cur_pos])
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
                print("Logits: ", logits)
                print("Logits shape: ", logits.shape)
                # return
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            print("next_token: ", next_token)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            print("next_token: ", next_token)
            print("res token before: ", tokens)
            tokens[:, cur_pos] = next_token
            print("res token after: ", tokens)
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            print("EOS REACHED: ", eos_reached)
            if all(eos_reached):
                break
            # if cur_pos == 11:
            #     return

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token