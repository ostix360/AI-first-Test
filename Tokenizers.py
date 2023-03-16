import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Encode text
text = "Hello, les gars mon chien est trop classe !"
tokens = tokenizer.tokenize(text)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)

inputs = torch.tensor([ids])

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

output = model(inputs)

print(output.logits)
