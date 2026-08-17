# Intro
Have thee ever woke up and decided thee wanted to write your own Shakespearean play? Well, I did! However, I realized that I am not an English Studies major, so I decided to learn transformer architecture from scratch and train my own model to do it instead. I've scrapped all Shakespear plays from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt and trained a decoder-only, causal Transformer language model. It includes
1. Positional embedding
2. Multi-head causal self-attention
3. Feedforward layers
4. Residual connection
5. Layer normalization
6. Dropout

# Sample Generation
```
ROMEO:
Why, but your butcher lament thee!

GLOUCESTER:
My lord, peters, do you will.

QUEEN ELIZABETH:
What lament hope,
I shall die guests with tongues?

HENRY BOLINGBROKE:
My lord says I do bid this traitor is prince?
```

# Scripts Explained
1. `bigram.py` has everything that's needed to train the model. It defines model architecture, and provides many nobs to tune. The parameters, such as `learning_rate`, `max_iters`, and `eval_interval` are defined at the top of the file. After training has been completed, it also saves the model weights and information using `torch.save()` To train the model, simply run 
```bash
python3 generate.py
```
2. `generate.py` provides a way to run inference on our trained model. It loads in `model.pt` from the specified directory(default is the current directory), and auto-regressively generate responses. 
```bash
python3 generate.py
```
Generates response with default inputs, aka no context and 500 tokens as the limit.

```bash
python3 generate.py --prompt="ROMEO:" --max-new-tokens=2000 > output-romeo.txt
```
Generates a 2000 token response with a starter context of "ROMEO:". The result of this generation is stored in `output-romeo.txt`.

3. `gpt-dev.ipynb` my learning sadbox. You can see me play with different things along the way! 

4. `input.txt` the downloaded Shakesphere and used as training data 

5. `output1.txt` generated output using command `python3 generate.py --max-new-tokens=2000 > output1.txt`

6. `output-juliet.txt` generated output using command `python3 generate.py --prompt="JULIET:" --max-new-tokens=2000 > output-juliet.txt`

7. `output-romeo.txt` generated output using ocmmand `python3 generate.py --prompt="ROMEO:" --max-new-tokens=2000 > output-romeo.txt`

# Interation Process and Progress at Each Step
### After implementing Single-head attention
Commit Hash: [`a8c958979b39d3acff09d57dbf08a205bab539c3`](https://github.com/YanfuOu/gpt-from-scratch/commit/a8c958979b39d3acff09d57dbf08a205bab539c3)
With 1 single head of attention, the current output looks like this: 
```
Whent whitridcowinen is by bth

Hiset bobe toe.
S:
O:
I thealilanss:
Want he uw hat vet?
F dilas ate
```
Loss:
step 4800: train loss 2.3798, val loss 2.4033

Looks slightly better than before, which was almost just randomness. However, we can do a better job using multihead attention! 

### After implementing Multi-head attention
Commit hash: [`f8861c611f38e82398223cb2ead02475fc2e7be2`](https://github.com/YanfuOu/gpt-from-scratch/commit/f8861c611f38e82398223cb2ead02475fc2e7be2#diff-183ef79fe759403d45b96c87142f4d991ab0ac5b3694cb146533c6bbc9d445b9)
After implementing 4 heads of self-attention, the current ouput looks like this: 
```
Wher?

RROMNOLORENCTAOLOLESHRKER:
Peak obe to tavegrtand that tands:
Waith fuus hat vet?
Fedilthoate
```
step 4800: train loss 2.2319, val loss 2.2670

Looks slightly better than before, and you can kind of pick out some words. Val loss now down to 2.27 from 2.40, which is a lot better! It helps to have multiple communication channels because these tokens have a lot to talk about. For example, they wanted to find the consonants, the vowls, or vowls from certain positions. Helps to create multiple independent channels of communication, gather lots of different types of data, and gather the output. 

### After implementing Feedforward Network
Commit hash: [`38e30169bc5c3ca01ff10a8e4b3e1d3f98a90f81`](https://github.com/YanfuOu/gpt-from-scratch/commit/38e30169bc5c3ca01ff10a8e4b3e1d3f98a90f81)
After implementing a standard 2-layer MLP, the current output looks like this:
```
Afet if bridcowe,
This by be madisen bube toe.
Sthe-' my dagieanss:
Warthie us him totbar dilacomoe
```
step 4800: train loss 2.2105, val loss 2.2290

Looks better than before by a bit. Val now down to 2.23 from 2.27, which is pretty good improvement! It helps to have feedforward network because it allows the tokens to think after communication(self-attention). The ReLu seperate the FeedForward layers and allows each layer to derive its meaning. 

### After implementing transformer block
Commit hash: [`13c1a313de3285b27e74a7f5c31ecbe8f5974153`](https://github.com/YanfuOu/gpt-from-scratch/commit/13c1a313de3285b27e74a7f5c31ecbe8f5974153)
After combining self-attention and feedforward block into a single repeatable transformer block, and training with 4 transformer blocks: 
```
Whent if try cowind, is sorst mas set bobe dowtarth ther mealceanss:
Want he uw crorvet?

MIXlassate
```
step 4800: train loss 2.3298, val loss 2.3565

Doesn't give a good result. If anything, it actually increased the tran and val losses! Why? We're starting to build a pretty deep neural network, and they suffer from optimization issues. There are 2 optimizations that can help with the depth of the network and ensure that they remain optimizable 
1. Skip/residual connections(the "Add block")
  - That means you have transformed the data, but still have a skip/residual connection from the previous features
2. Layer normalization(the "Norm block")

### After implementing Residual Connection
Commit hash: [`233c369e58c88b881fa2c46ef9816b05540019b9`](https://github.com/YanfuOu/gpt-from-scratch/commit/233c369e58c88b881fa2c46ef9816b05540019b9)
After implementing residual connection from the "Attention is All You Need" paper:
```
Cle beford
Thow and Ours, be madient bube toe.
Sagraves me?

Tauspuar bache us he hert?
Wedilthoate
```

step 4800: train loss 1.9866, val loss 2.0779
We can see that the tran loss is getting ahead of the val loss. This means we're seeing a bit of overfitting. Our generation isn't amazing, but we can make out different words, like "throw", "and", "ours" etc. Our val loss is down to 2.077, which is pretty good! 

### After implementing LayerNorm
Commit hash: [`eb4808f7c9d7fda2091b1c98174844194963b9c5`](https://github.com/YanfuOu/gpt-from-scratch/commit/eb4808f7c9d7fda2091b1c98174844194963b9c5)
After implementing Layernorm in self-attention blocks:
```
JULIO:
Reridce.

SOROMET:
He madise, bube to take O-dam the alause:
Waith foul he hert?
Fedinghoate
```
step 4800: train loss 1.9758, val loss 2.0676

You can see a slight improvement in val loss down to 2.06 by adding the layernorm. This will help more when we have bigger and deeper networks.

### After adding Dropout and scaled the model wayyyyy upppp
Commit hash: [`edbab5756c10367ad69c4402088379dc67216da6`](https://github.com/YanfuOu/gpt-from-scratch/commit/edbab5756c10367ad69c4402088379dc67216da6)
Scaled:
1. block_size 8 --> 256
2. batch_size 32 --> 64
3. learning_rate 1e-3 --> 3e-4
4. n_embed 32 --> 384
5. n_head 4 --> 6
5. n_layer 1 --> 6
6. drop_out = 0.2
```
Go as if that Angelo. Thou know'st; there well.

BLAnday:
There's your leave as kight o' the house, 
```
step 4800: train loss 1.0624, val loss 1.5138

Now that sounded like sweet Shakespere!