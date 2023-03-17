import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

checkpoint = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


raw_datasets = load_dataset("paws-x", "fr")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print(raw_datasets)

# sequences = [                                    TODO : set Env var HF_HOME=/mnt/d/DataScience/HG_cache
#     "J'ai attendu toute ma vie un cours sur HuggingFace.",
#     "Ce cours est incroyable !",
#     "Malgrès cela je ne comprends pas comment on peut utiliser un modèle de classification pour faire de la traduction."
# ]
#
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
#
# batch["labels"] = torch.tensor([1, 1, 1])
#
# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()

