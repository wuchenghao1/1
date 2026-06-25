# 中文情感分析示例项目

这是一个简单的情感分析（二分类）示例项目，使用 Hugging Face Transformers + datasets，支持从公开数据集下载（如 imdb），也可以加载本地 CSV 数据。

目录结构：
- src/
  - data.py     # 数据加载与分词
  - model.py    # 模型构建
  - train.py    # 训练脚本
  - infer.py    # 推理脚本
- requirements.txt
- README.md

快速开始（用 CPU 或 GPU 都可）：
1. 安装依赖：
   pip install -r requirements.txt
2. 直接训练（会自动下载 imdb 数据集）：
   python src/train.py --output_dir outputs --model_name_or_path distilbert-base-uncased --dataset imdb --epochs 1 --batch_size 16
3. 推理示例：
   python src/infer.py --model_dir outputs --text "This movie was fantastic!"

如果你希望我改为使用本地数据文件（CSV），把 CSV 上传到仓库或告诉我文件路径格式，我会调整脚本或直接把数据加载脚本改成更适合你的数据格式。