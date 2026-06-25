# 中文情感分析示例项目

这是一个简单的情感分析（二分类）示例项目，使用 Hugging Face Transformers + datasets，支持从公开数据集下载（如 imdb），也可以加载本地 CSV 数据。

新增内容：
- data/sample.csv — 小型示例数据（中英混合），方便本地快速跑通训练/推理流程
- notebooks/train_demo.ipynb — Notebook 演示如何用 sample.csv 快速训练并推理

目录结构：
- data/sample.csv  # 小型示例数据（text,label）
- src/
  - data.py     # 数据加载与分词
  - model.py    # 模型构建
  - train.py    # 训练脚本
  - infer.py    # 推理脚本
- requirements.txt
- README.md

快速开始（用 CPU 或 GPU 都可）：
1. 克隆并进入仓库：
   git clone https://github.com/wuchenghao1/1
   cd 1
2. 安装依赖：
   pip install -r requirements.txt
3. 用示例数据训练（1 epoch）：
   python src/train.py --local_csv data/sample.csv --model_name_or_path distilbert-base-uncased --output_dir outputs_sample --epochs 1 --batch_size 8
4. 推理示例：
   python src/infer.py --model_dir outputs_sample --text "This movie was fantastic!"

如果你有自己的 CSV 数据，请确保包含列 `text` 和 `label`（整数标签，例如 0/1），然后用 --local_csv 指向你的文件路径。

下一步的建议（我可以替你继续）：
- 使用中文预训练模型（如 bert-base-chinese 或 hfl/chinese-roberta-wwm-ext）并用中文数据微调
- 加入模型评估可视化（混淆矩阵、分类报告）并保存训练日志
- 支持更复杂的训练策略（early-stopping, LR scheduler, mixed precision, accelerate）
- 如果你希望我继续，我会基于仓库更新这些改进并提交。