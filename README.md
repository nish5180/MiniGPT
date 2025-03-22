# MiniGPT
Course Project for 'Bayesian Machine Learning'

Things to focus on:
1. Try out different tokenisers and see how the output changes: Sentence Tokenisers, BPE tokenisers
2. Attention plots
3. Model architecture- different feed forward models
4. Decoding schemes?


Ideas:

1. Your Own Tokenizer Engine
Build a custom tokenizer class from scratch (char-level + optional BPE).

Implement your own train(), encode(), decode().

Add visualizations: show how "hello world" gets tokenized step-by-step.


2. Train a "Language Style Transfer" GPT
Train on Shakespeare → fine-tune on modern text (e.g., Reddit comments, tweets).

Compare how the same prompt sounds before and after fine-tuning.


3. "Explain Your Prediction" Mode
At each generation step, print the top 3 most attended tokens.

Like:
"Model predicted 'mat' because it paid attention to 'sat' and 'on'"


4. Your Own Minimal Attention Head
Don’t use PyTorch’s nn.MultiheadAttention. Write it completely from scratch with just matmuls and softmax.

Visualize the Q, K, V matrices and attention scores on a sample.


5. Build a Training Playground
Let the user toggle:

Number of layers

Block size

Attention head count

Plot loss curves and generation examples for each setup.

