#!/usr/bin/env python
# 编码: utf-8

# # 评估

# ### 在 GPU 上运行此代码的步骤非常少，适用于其他环境（例如在 Lamini 上）。
# ```
# finetuned_model = BasicModelRunner(
#     "lamini/lamini_docs_finetuned"
# )
# finetuned_output = finetuned_model(
#     test_dataset_list # 分批处理!
# )
# ```
#
# ### 让我们再次看看内部实现！这是 Lamini 的 `llama` 库的开放核心代码 :)

# 导入必要的库和模块
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd

import transformers
import datasets
import torch
# 用于生成进度条，当你在 Python 中运行需要一些时间完成的循环或任何迭代操作时，tqdm 可以帮助你可视化操作的进度。
from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
global_config = None


# 加载数据集
dataset = datasets.load_dataset("lamini/lamini_docs")
test_dataset = dataset["test"]

# 打印测试数据集中的问题和答案
print(test_dataset[0]["question"])
print(test_dataset[0]["answer"])

# 设置模型和分词器
model_name = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# ### 设立一个非常基本的评估函数
# 精确匹配函数
def is_exact_match(a, b):
    return a.strip() == b.strip()

# 设置模型为"评估模式"
# 这主要涉及两个方面：
# 1. 关闭模型的学习阶段：当模型处于评估模式时，所有与训练相关的操作，如参数更新（梯度更新），都会被停用。
#    这确保模型在评估期间的表现仅与其学习到的知识有关，而不会因为新数据而改变。
# 2. 规范化层行为的改变：对于许多类型的网络，尤其是包含批量归一化（Batch Normalization）和丢弃（Dropout）层的网络，其行为在训练和评估时是不同的。
#    例如：批量归一化（Batch Normalization）：在训练模式下，这一层会对每个批次的数据计算均值和方差，并用这些统计量进行归一化。在评估模式下，批量归一化会使用整个训练集的均值和方差来进行归一化，这些值在训练过程中已经计算好并固定下来。
#         丢弃（Dropout）：在训练模式下，丢弃层会随机地使网络中的某些神经元的输出为0，以防止模型过拟合。在评估模式下，所有神经元都会被保留，但其输出会相应地进行缩放，确保输出的总体活性与训练时相同。
# model.eval()主要用于确保模型在进行验证或测试时能够稳定地表现出其真实的性能，不受训练过程中引入的随机性和某些特定层行为的影响。
model.eval()


# 推理函数定义
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # 分词
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.encode(
      text,
      return_tensors="pt",
      truncation=True,
      max_length=max_input_tokens
  )
  # 生成
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )
  # 解码
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
  # 移除提示文本
  generated_text_answer = generated_text_with_prompt[0][len(text):]
  return generated_text_answer


# ### 运行模型并与预期答案比较

# 测试和生成答案（）
test_question = test_dataset[0]["question"]
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)

# 打印答案
answer = test_dataset[0]["answer"]
print(answer)

# 精确匹配检查，一般情况下下很难100%精准匹配上。
# 我们可以借助第三方LLM来测试评估模型回答的内容是否符合。
# 也可以使用向量相似度进行检测。
exact_match = is_exact_match(generated_answer, answer)
print(exact_match)



# ### 在整个数据集上运行

# 在测试数据集的前n项上运行模型并记录精确匹配结果
n = 10
metrics = {'exact_matches': []}
predictions = []
# 循环并显示进度条
for i, item in tqdm(enumerate(test_dataset)):
    print("i Evaluating: " + str(item))
    question = item['question']
    answer = item['answer']

    try:
      predicted_answer = inference(question, model, tokenizer)
    except:
      continue
    predictions.append([predicted_answer, answer])
    # 修正：精确匹配检查
    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    if i > n and n != -1:
      break
print('Number of exact matches: ', sum(metrics['exact_matches']))


# 将预测结果和目标答案保存到DataFrame
# 数据帧（DataFrame）是一种在数据科学和统计分析中广泛使用的数据结构，主要用于存储和操作结构化数据。
# 数据存储，数据操作，数据分析，数据清洗，数据可视化等
df = pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])
print(df)


# ### 评估所有数据

# 加载评估数据集
evaluation_dataset_path = "lamini/lamini_docs_evaluation"
evaluation_dataset = datasets.load_dataset(evaluation_dataset_path)

# 将评估数据集保存到DataFrame
pd.DataFrame(evaluation_dataset)

# ### 尝试 ARC 基准测试
# 这可能需要几分钟
# 运行 ARC 基准测试
# !python lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=lamini/lamini_docs_finetuned --tasks arc_easy --device cpu