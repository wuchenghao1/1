# 中文情感分析示例项目

这是一个简单的情感分析（二分类）示例项目，使用 Hugging Face Transformers + datasets，支持从公开数据集下载（如 imdb），也可以加载本地 CSV 数据。

改动说明：
- 默认预训练模型已改为 hfl/chinese-roberta-wwm-ext（更适合中文情感数据）。
- 训练完成后会对测试集做一次完整评估，并把下面两个 artifact 保存在输出目录：
  - confusion_matrix.png — 混淆矩阵图片
  - classification_report.csv — sklearn 的分类报告（precision/recall/f1）

新增内容：
- data/sample.csv — 小型示例数据（中英混合），方便本地快速跑通训练/推理流程
- notebooks/train_demo.ipynb — Notebook 演示如何用 sample.csv 快速训练并推理

快速开始（用 CPU 或 GPU 都可）：
1. 克隆并进入仓库：
   git clone https://github.com/wuchenghao1/1
   cd 1
2. 安装依赖：
   pip install -r requirements.txt
3. 用示例数据训练（1 epoch）：
   python src/train.py --local_csv data/sample.csv --model_name_or_path hfl/chinese-roberta-wwm-ext --output_dir outputs_sample --epochs 1 --batch_size 8

训练完成后会在 outputs_sample 中生成：
- pytorch 模型文件（供 infer.py 使用）
- confusion_matrix.png
- classification_report.csv

如果你有自己的 CSV 数据，请确保包含列 `text` 和 `label`（整数标签，例如 0/1），然后用 --local_csv 指向你的文件路径。

下一步（可选）：
- 我可以把 notebook 里直接运行的模型名也改为中文并演示一轮训练/评估（并把输出截图提交），或者我可以添加 GitHub Actions 做自动 smoke-test。
