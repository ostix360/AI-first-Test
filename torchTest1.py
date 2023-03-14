from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel
import torch

classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
classifier('We are very happy to show you the 🤗 Transformers library.')

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw = [
    "I'm very happy to show you the 🤗 Transformers library.",
    # "fqfbquifbquibff"
]
inputs = tokenizer(raw, padding=False, truncation=False, return_tensors="pt")
print(inputs)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs)

prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(prediction)
var = model.config.id2label
print(var)