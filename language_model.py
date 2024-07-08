import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#open dataset and sort chars
with open('NEWS.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
print(chars)
vocabulary_size = len(chars)

#tokenize
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100])
