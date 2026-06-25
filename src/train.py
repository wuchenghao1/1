import argparse
import os

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import ClassLabel

from src.data import load_and_tokenize
from src.model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='distilbert-base-uncased')
    parser.add_argument('--dataset', type=str, default='imdb', help='Hugging Face dataset name or "local" for CSV')
    parser.add_argument('--local_csv', type=str, default=None, help='本地 CSV 路径（如果使用本地数据）')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--sample_size', type=int, default=None, help='仅用于快速调试，设置训练样本数')
    args = parser.parse_args()

    dataset_name = None if args.local_csv else args.dataset
    ds, tokenizer = load_and_tokenize(dataset_name, local_csv=args.local_csv, model_name=args.model_name_or_path, max_length=args.max_length, sample_size=args.sample_size)

    # 尝试获取标签数
    if 'label' in ds['train'].features:
        labels_feature = ds['train'].features['label']
        if isinstance(labels_feature, ClassLabel):
            num_labels = labels_feature.num_classes
        else:
            # 假设二分类
            num_labels = 2
    else:
        num_labels = 2

    model = create_model(args.model_name_or_path, num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2,
    )

    def compute_metrics(p):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted')
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()
