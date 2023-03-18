import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, BertModel, DataCollatorWithPadding
from datasets import load_dataset

# Même chose que précédemment
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#model = BertModel.from_pretrained(checkpoint)

#print(torch.cuda.is_available())


def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


raw_datasets = load_dataset("glue", "mrpc")

# print(raw_datasets)

raw_train_data = raw_datasets["train"]
raw_validation_data = raw_datasets["validation"]

#tokenized_sentence1 = tokenizer(raw_train_data[15]["sentence1"], raw_train_data[15]["sentence2"], padding=True,
#                                truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_colator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])   # print the length of each sample max length is 67
batch = data_colator(samples)                # collate the samples into a batch with the maximum length of 67
print({k: v.shape for k, v in batch.items()})  # print the shape of each tensor in the batch 


# sequences = [                                  TODO : set Env var HF_HOME=/mnt/d/DataScience/HG_cache
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
#
# # Ceci est nouveau
# batch["labels"] = torch.tensor([1, 1])
#
# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()
