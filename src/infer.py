import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def predict(text: str, model_dir: str = 'outputs', device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label = int(probs.argmax())
    return label, probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='outputs')
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()

    label, probs = predict(args.text, model_dir=args.model_dir)
    print('Predicted label:', label)
    print('Probabilities:', probs)

if __name__ == '__main__':
    main()
