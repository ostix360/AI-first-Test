def datasets_post_process(tokenized_dataset):
    final_datasets = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
    final_datasets = final_datasets.rename_column("label", "labels")
    final_datasets.set_format(type="torch")
    # Debug
    print(final_datasets["train"].column_names)
    return final_datasets


def debug_data_processing(train_dataloader):
    batch = None
    for batch in train_dataloader:
        break
    print({k: v.shape for k, v in batch.items()})
    return batch
