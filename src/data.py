# 自动生成或加载数据集并进行分词
from typing import Optional, Tuple

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def load_and_tokenize(dataset_name: Optional[str] = "imdb", local_csv: Optional[str] = None, model_name: str = "distilbert-base-uncased", max_length: int = 256, sample_size: Optional[int] = None) -> Tuple[DatasetDict, AutoTokenizer]:
    """加载数据集并返回 tokenized 的 DatasetDict 和 tokenizer。

    如果提供 local_csv，会尝试用 pandas 读取 csv，期待列 ['text','label']。
    否则从 Hugging Face datasets 下载 dataset_name（例如 imdb 或 yelp_polarity）。
    sample_size 可用于快速调试（从训练集抽取小样本）。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if local_csv:
        import pandas as pd
        df = pd.read_csv(local_csv)
        assert 'text' in df.columns and 'label' in df.columns, "local CSV must have 'text' and 'label' columns"
        ds = DatasetDict()
        from datasets import Dataset
        DS = Dataset.from_pandas(df)
        # 简单拆分
        train_test = DS.train_test_split(test_size=0.1)
        ds['train'] = train_test['train']
        ds['test'] = train_test['test']
    else:
        ds = load_dataset(dataset_name)
        # 有些数据集是 train/test split already; 如果没有 validation，尝试切分
        if 'validation' not in ds:
            ds = ds.train_test_split(test_size=0.1)

    if sample_size:
        ds['train'] = ds['train'].select(range(min(sample_size, len(ds['train']))))

    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)

    ds = ds.map(preprocess, batched=True)
    return ds, tokenizer
