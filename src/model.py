from transformers import AutoModelForSequenceClassification


def create_model(model_name_or_path: str = "distilbert-base-uncased", num_labels: int = 2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    return model
