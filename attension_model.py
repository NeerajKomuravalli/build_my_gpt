import torch
import torch.nn as nn

from torch.nn import functional as F
from tqdm import tqdm

from utils import read_data, get_batch
from encoder_decoder import EncoderDecoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    """Attension head definition"""
    def __init__(self, embedding_size:int, block_size:int, head_size:int):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x => [B, T, C]
        B, T, C = x.shape
        k = self.key(x) # [B, T, head_size]
        q = self.query(x) # [B, T, head_size]
        v = self.value(x) # [B, T, head_size]

        weight = q @ k.transpose(-2, -1) * C**-0.5 # [B, T, T]
        weight = torch.masked_fill(weight, self.tril[:T, :T] == 0, -1*torch.inf)
        weight = torch.softmax(weight, dim=-1) # [B, T, T]
        weight = self.dropout(weight)

        out = weight @ v # [B, T, head_size]

        return out

class MultiHead(nn.Module):

    def __init__(self, no_of_heads:int, embedding_size:int, block_size:int, head_size:int):
        super().__init__()
        self.heads = nn.ModuleList([Head(embedding_size, block_size, head_size) for _ in range(no_of_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):

    def __init__(self, embedding_size:int, dropout:float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size*4),
            nn.ReLU(),
            nn.Linear(embedding_size*4, embedding_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # x => [B, T, embedding_size]
        out = self.net(x)

        return out
    
class Block(nn.Module):

    def __init__(self, no_of_heads:int, embedding_size:int, block_size:int, head_size:int, dropout:float):
        super().__init__()
        self.multi_head = MultiHead(no_of_heads, embedding_size, block_size, head_size)
        self.feed_forward = FeedForward(embedding_size, dropout)

    def forward(self, x):

        out = self.multi_head(x)
        out = self.feed_forward(x)

        return out
    
class TransformerDecoderModel(nn.Module):

    def __init__(self, vocab_size:int, no_of_heads:int, embedding_size:int, block_size:int, head_size:int, n_layer:int ,dropout:float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoder = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(no_of_heads, embedding_size, block_size, head_size, dropout) for _ in range(n_layer)])
        self.proj = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, target=None):
        B, T = x.shape

        emb = self.embedding(x) #[B, T, embedding_size]
        pos = self.positional_encoder(torch.arange(T, device=device)) #[T, embedding_size]
        out = emb + pos #[B, T, embedding_size]
        out = self.blocks(out) #[B, T, embedding_size]
        logits = self.proj(out)

        loss = None
        if target is not None:
            # target => [B, T]
            B, T, vocab_size = logits.shape
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        
        return logits, loss

    def generate(self, x, max_new_tokens:int=100):
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits, _ = self(x_cond) # [B, T, vocab_size]
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            next_x = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_x), dim=1)

        return x

if __name__ == "__main__":
    data_path = "../data/tinyshakespeare.txt"
    data = read_data(data_path)
    vocab = list(set(data))
    encoder_decoder = EncoderDecoder(vocab)
    encoded_data = encoder_decoder.encode(data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_tensor = torch.tensor(encoded_data, dtype=torch.long)

    # head_size * no_of_heads = embedding_size
    vocab_size = len(vocab)
    no_of_heads = 8
    embedding_size = 64
    head_size = embedding_size // no_of_heads
    block_size = 128
    n_layer = 6
    dropout = 0.2

    transformer = TransformerDecoderModel(vocab_size, no_of_heads, embedding_size, block_size, head_size, n_layer,dropout)
    transformer = transformer.to(device)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-4)

    batch_size = 64
    train_steps = 5000

    # Wrap the training loop with tqdm for progress tracking
    for _ in tqdm(range(train_steps), desc="Training Progress"):
        x, y = get_batch(block_size, batch_size, data)
        _, loss = transformer(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Final Loss : {loss.item():.4f}")

    start = torch.tensor(encoder_decoder.encode("\n"), dtype=torch.long, device=device)
    start = start.view(1, 1)

    print(encoder_decoder.decode(transformer.generate(start, max_new_tokens=500)[0].tolist()))