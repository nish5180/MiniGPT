import tiktoken

tokenizer = tiktoken_lib.get_encoding("gpt2")

text = "To be, or not to be, that is the question."

tokens = tokenizer.encode(text)
print("Original Text:\n", text)
print("\nToken IDs:\n", tokens)

decoded_text = tokenizer.decode(tokens)
print("\nDecoded Text:\n", decoded_text)

# Token count
print("\nNumber of tokens:", len(tokens))
