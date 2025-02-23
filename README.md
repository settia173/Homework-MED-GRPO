# MedQwen3B-Reasoner
MedQwen3B-Reasoner 是一个专门针对医学领域推理和数学问题求解优化的3B参数语言模型。该模型基于Qwen2.5-3B-Instruct，通过GRPO (Group Relative Policy Optimization) 进行了医学领域的特定适配。
# 参考：
https://medium.com/@hooman_66365/build-your-own-medical-mini-deepseek-r1-with-reinforcement-learning-508509cd7d83
# HUGGINGFACE仓库：
https://huggingface.co/AEONA/QWEN-MED-GRPO-AEONA
## 主要特点

- 结构化推理输出：使用 `<reasoning>`/`<answer>` 标签格式
- 4-bit 量化部署：通过 unsloth 实现高效推理
- 多领域训练数据：
  - PubMedQA (70%)
  - GSM8K 数学推理数据集
  - Health Benchmarks 医疗多选题数据集

## 模型训练

模型使用以下技术进行训练：

- GRPO (Group Relative Policy Optimization) 用于领域适配
- LoRA (Low-Rank Adaptation) 用于参数高效微调
- 4-bit 量化以提高训练和推理效率
- vLLM 用于加速推理

## 训练数据集

模型使用了以下数据集进行训练：
- `openai/gsm8k`: 数学推理问题
- `qiaojin/PubMedQA`: 医学文献问答
- `yesilhealth/Health_Benchmarks`: 医疗多选题

## 用gradio做前端
![2025-02-23 12-35-57屏幕截图](https://github.com/user-attachments/assets/9a231473-6e94-4d60-b1d4-c567cbf48149)
![2025-02-23 12-34-20屏幕截图](https://github.com/user-attachments/assets/cdbc21d1-971a-43ca-91f6-7a31a2858580)

  
