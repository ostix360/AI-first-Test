from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Encode text
text = "Hello, les gars mon chien est trop classe !"
tokens = tokenizer.tokenize(text)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)

tokenizer.save_pretrained("tokenizer")
