# 大模型复现任务
环境在wsl中配置，conda 下的llm
安装第三方包

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers accelerate datasets
pip install gradio

## 包名	用途
### 深度学习相关
pytorch	核心深度学习框架，支持张量计算、模型训练与推理
torchvision	图像相关工具包（主要用于 CV，如模型、数据集、转换等）
torchaudio	音频处理工具包（如语音识别任务）
pytorch-cuda=12.1	安装与 GPU CUDA 驱动匹配的加速后端，使 PyTorch 使用你的 RTX 4070 进行推理
### huggingface相关
transformers	HuggingFace 的主力库，用于加载、运行、微调各种预训练大语言模型（如 Qwen、Mistral、BERT 等）
accelerate	自动配置 PyTorch 的多卡/多设备训练推理环境（本地或云上）
datasets	统一的数据集加载框架，可以访问上万个 NLP/CV 数据集（如 SQuAD、COCO、AG News 等）
### 页面交互用
pip install gradio
## 额外
sentencepiece	用于加载一些模型（如 LLama、Qwen）使用的 tokenizer
einops	一些模型（如 Vision Transformer、多模态模型）需要它进行张量操作
大模型推理更省内存	bitsandbytes	低比特量化推理（4bit/8bit）
可视化	matplotlib / seaborn	做图、画 attention 可视化
Web 部署	fastapi, uvicorn	如果你想做 API 而不是界面

# 使用教程
## 调用程序


模块名	功能职责
models/	本地预训练模型存储路径
llm/	模型的加载 + 推理 + Prompt 工程
rag/	检索相关逻辑（如 Milvus, FAISS 等）
ui/	基于 Gradio/Web 的交互界面
utils/	工具类、通用代码、日志封装
tests/	模块测试脚本
config/	模型配置文件，例如温度、top_p 等参数