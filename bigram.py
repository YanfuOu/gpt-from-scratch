import torch 
import torch.nn as nn 
from torch.nn import functional as F

block_size = 8 # how big is each sample
batch_size = 32 # how many sample to draw
eval_iters = 200 # during estimate loss, how many batches do we use
learning_rate = 1e-3 # how big are the steps we take during training using back prop
eval_interval = 300 # interval in which we stop and calculate loss 
max_iters = 5000 # how many times to run the training loop
n_embed = 32 # short for number of embedding dims
# head_size = 16 # defines how big each attention head is

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# train test splits 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data 
    index = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in index])
    y = torch.stack([data[i+1:i+block_size+1] for i in index])
    x, y = x.to(device), y.to(device)
    return x, y

# get less noisey loss by getting the average loss over eval_iters number of batches for 
# both train and test splits 
# optimization that tells pytorch that everything inside this function won't call .backwards
# so pytorch can be more efficient 
@torch.no_grad() 
def estimate_loss():
    out = {}
    model.eval() # setting model to be in evaluation phase 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # initialize this tensor, which will store all the losses 
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y) # evaluate loss from the model
            losses[k] = loss.item() 
        out[split] = losses.mean()
    model.train() # setting the model back in training phase 
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Tril isn't a variable in Pytorch, it's a variable. assigned to the module using register_buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # self-attention part that only looks back --> masking with tril
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        # performs the weighted aggregation of the values 
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out 
        
        

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # this encodes the identity of the tokens
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        # initialize the 1 head of self-attention first and name it "self-attention head"
        # choosing the keep the head size as n_embed 
        self.sa_head = Head(n_embed)

        # We don't want to go directly from embedding to logits  
        # to go from token embedding to logits, we are adding a linear layer
        self.lm_head = nn.Linear(n_embed, vocab_size) # lm_head = language model head 

        # we also want to encode the position of the tokens 
        # each position from 0 to block_size - 1 will also get its own embedding vector 
        self.position_embedding_table = nn.Embedding(block_size, n_embed) 

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # each forward pass computes prediction for idx(input), 
        # and loss, which is how far that prediction is from the target
        tok_emb = self.token_embedding_table(idx)

        # these are basically integers from 0 to t-1, they all get embedded into the table to create (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        # we get the logits from passing our token_emb through lm_head

        # now x not only holds the token identities, but also the positions at which these tokens occur
        x = tok_emb + pos_emb # (B, T, C)

        # now compute that 1 head of self-attention after encoding token + position embeddings 
        x = self.sa_head(x)
        # now output goes to the decoder and create the logits 
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
             # all the T timesteps(i.e. 1st elem of the B(batch), 2nd elem of the B(batch) ) 
             # across all batches basically becomes a n x 1 vector
             # C is the channel -> contains the prob of a character to all other chars 
             # from the embedding table
            logits = logits.view(B*T, C)
            # B * T is same idea as above, except now you're storing the target, so no need for C
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss 

    def generate(self, idx, max_new_tokens):
        # idx is the starter token 
        for i in range(max_new_tokens):
            # now with self-attention head, we need to make sure that our idx 
            # that we feed into the model can never have more than block size
            # now that we use positional embedding, leading to our positional embedding
            # table to run out of scope, since it only has embeddings up till block size
            idx_cond = idx[:, -block_size:] # crop the context that we feed into self

            logits, loss = self(idx_cond)
            # focus on only the last timestep
            logits = logits[:,-1, :] # becomes (B, C)
            # create probility dist from logits distributions using softmax
            probs = F.softmax(logits, dim=-1)
            # draws from the prob distribution to get the token
            idx_next = torch.multinomial(probs, num_samples=1)
            # use that token to get next token
            idx = torch.cat((idx, idx_next), dim=1)
        return idx 

model = BigramLanguageModel() # initialize the model, which calls __init__
m = model.to(device) # loads the model into GPU, will also move the model weights
# in this case, it would be nn.Embedding 

# creates a PyTorch optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # evaluate the loss every once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device) # starter token
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))