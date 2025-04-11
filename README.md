# README

This repository builds upon [Andrej Karpathy’s GPT](https://github.com/karpathy/nanoGPT) with the goal of extending and customizing the architecture to explore tokenizer design, hyperparameter optimization, and encoder-decoder modifications.

## Project Goals

- **Tokenizer Comparison**  
  Evaluate different tokenization strategies including:
  - Character-level encoding
  - Custom Byte Pair Encoding (BPE)
  - Built-in `tiktoken` BPE tokenizer

- **Hyperparameter Optimization**  
  Apply **Bayesian optimization** using Optuna to fine-tune training settings.

- **Encoder Addition**  
  Modify the architecture to include an encoder. While the decoder is trained on Shakespeare's works (as in the original), the encoder is trained on modern pop music lyrics.  
  **Note:** One challenge in this setup is the **lack of a suitable validation set** for the encoder-decoder configuration.

## Dependencies

Install all required packages with:

- `torch` — PyTorch for model training  
- `tiktoken` — Tokenization library  
- `optuna` — Bayesian hyperparameter optimization

## How to Run

1. **Data Preparation**  
   (Optional for the decoder run)Run `generate_pairs.py` to create a CSV file of data apairs for the encoder input and decoder output.  
   The Shakespeare dataset is loaded automatically in the relevant scripts and does not require preprocessing.

2. **Model Training and Testing**
   - `gpt.py`: Baseline GPT model without encoder, used for initial testing, using Shakespeare data ('input.txt')
   - `hyperparameter_search.py`: Performs standalone hyperparameter optimization using Shakespeare data (`input.txt`).
   - `encoder.py`: Extended version of the GPT model with a full encoder architecture.  
     Takes pairwise data ('pair dataset.csv') and outputs generated text to `generate.txt`.

## Model Sizes

Initial experiments and architectural testing are conducted on smaller-scale models which is 'scaled_down_GPT.py' which has 0.38M parameters.

##Sample of the output/ generated text:


BRUTUS:
By his present power,

We raise not to his visage all:

just, be that may

give him the rest, who is ground

to see you complies,

For it is of out of.



MARCIUS:

Three could be

Our word in their mar’s safety,

when he hath ever run

For can seem attainous tender that title:

till Hath made him has him, he many child,

And would anger him the painted shadow of his helm,

And merited a sweets of liking is too.


