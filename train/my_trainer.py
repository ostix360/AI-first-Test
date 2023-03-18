from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

import utils

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

final_tokenized_datasets = utils.datasets_post_process(tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    final_tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    final_tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

batch = utils.debug_data_processing(train_dataloader)

outputs = model(**batch)

# Debug
print(outputs.loss, outputs.logits.shape)

