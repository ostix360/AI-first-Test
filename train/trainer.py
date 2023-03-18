from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, \
    Trainer
import evaluate
import numpy as np

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


def compute_metrics(eval_pred):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# def limit_train_dataset(dataset, limit):
#     if limit is not None:
#         # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
#         dataset["input_ids"] = dataset["input_ids"][:limit]
#         dataset["attention_mask"] = dataset["attention_mask"][:limit]
#         dataset["label"] = dataset["label"][:limit]
#
#
#
# limit_train_dataset(raw_datasets["train"], 1000)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test_trainer", evaluation_strategy="steps", max_steps=100, save_steps=50)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# print(len(tokenized_datasets["train"]))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
#
# trainer.evaluate()
