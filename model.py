from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I'm very happy to show you the 🤗 Transformers library."

model_inputs = tokenizer(sequence)

# outputs = model(**model_inputs)
#
# print(outputs.logits)
print(tokenizer.decode(model_inputs["input_ids"]))
