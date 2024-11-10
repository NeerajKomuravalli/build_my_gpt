import os
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F


def read_data(file_path) -> str:
    if not os.path.exists(file_path):
        raise FileExistsError(f"{file_path} is not a valid path")
    data = []
    with open(file_path, "r") as f:
        data = f.readlines()
    
    data_str = "\n".join(data)

    return data_str

class EncoderDecoder:
    def __init__(self, vocab:List[str]):
        # Letter to number
        self.encoder_dict = {}
        # Number to letter
        self.decoder_dict = {}
        # Generate encoder and decoder dicts
        for i, char in enumerate(vocab):
            self.encoder_dict[char] = i
            self.decoder_dict[i] = char
        

    def encode(self, data: str) -> List:
        encoded_data = []
        for d in data:
            encoded_data.append(self.encoder_dict[d])
        
        return encoded_data

    def decode(self, data:List) -> str:
        decoded_data = ""
        for d in data:
            decoded_data += self.decoder_dict[d]

        return decoded_data

def get_batch(block_size, batch_size, data_tensor):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])

    return x, y

class BigramModel(nn.Module):
    def __init__(self, vocab_size:int):
        super().__init__()
        self.embeddings_layer = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logits = self.embeddings_layer(idx)

        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))

        return logits, loss

    def generate(self, idx, no_of_tokens:int):
        for _ in range(no_of_tokens):
            logits, loss = self(idx, None)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def train(model:BigramModel, optimizer, batch_size:int=32, train_steps:int=10000, block_size:int=8):

    for _ in range(train_steps):
        x, y = get_batch(block_size, batch_size, data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return loss

def generate_data(model:BigramModel, start_string="\n"):
    start = torch.tensor(encoder_decoder.encode("\n"), dtype=torch.long)
    start = start.view(1, 1)

    return encoder_decoder.decode(model.generate(start, no_of_tokens=100)[0].tolist())


if __name__ == "__main__":
    data_path = "../data/tinyshakespeare.txt"
    data = read_data(data_path)
    vocab = list(set(data))
    encoder_decoder = EncoderDecoder(vocab)
    encoded_data = encoder_decoder.encode(data)

    data_tensor = torch.tensor(encoded_data, dtype=torch.long)

    bigram = BigramModel(len(vocab))
    optimizer = torch.optim.AdamW(bigram.parameters(), lr=1e-3)
    batch_size:int=32
    train_steps:int=10000
    block_size:int=8

    loss = train(bigram, optimizer, batch_size, train_steps, block_size)

    print(generate_data(model=bigram))
