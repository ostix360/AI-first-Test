import torch
from transformers import BertConfig, BertModel, CamembertTokenizer, CamembertConfig, CamembertModel

print(torch.cuda.is_available())
sequences = ["Hello!", "Cool.", "Nice!"]


# Construire la configuration
config = CamembertConfig()

# Construire le modèle à partir de la configuration
model = CamembertModel(config)

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
encoded_sequences = tokenizer(sequences, return_tensors="pt")
print(encoded_sequences)

model_inputs = torch.tensor(encoded_sequences["input_ids"])
outputs = model(model_inputs)
print(outputs)