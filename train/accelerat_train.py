from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, BertTokenizer, BertModel

import utils

raw_datasets: DatasetDict = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
accelerator: Accelerator = Accelerator()
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(checkpoint)
model: BertModel = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def tokenize_function(examples: dict):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
final_tokenized_datasets = utils.datasets_post_process(tokenized_datasets)
data_collator: DataCollatorWithPadding = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    final_tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    final_tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Debug
# batch = utils.debug_data_processing(train_dataloader)
# outputs = model(**batch)
# print(outputs.loss, outputs.logits.shape)

optimizer = AdamW(model.parameters(), lr=3e-5)
a_train_dataloader, a_eval_dataloader, a_model, a_optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 1
lr_scheduler, num_training_steps = utils.get_lr_scheduler(a_optimizer, a_train_dataloader, num_epochs)

utils.train_with_accelerator(a_model, a_train_dataloader, num_epochs, a_optimizer, lr_scheduler, accelerator, num_training_steps)
evaluation = utils.evaluate_with_accelerate(model, a_eval_dataloader, accelerator)
print(evaluation)
