#!/usr/bin/env python
# coding: utf-8

# ## 微调数据：与预训练和基础准备比较
import jsonlines
import itertools
import pandas as pd
from pprint import pprint

# Hugging Face 提供的一个非常有用的 Python 库，专门用于加载和处理自然语言处理（NLP）数据集。
# load_dataset 函数是这个库的核心功能之一，它允许你轻松地加载来自Hugging Face的数据集库中的各种标准数据集。
# 这个库支持数百种不同的数据集，并提供了一种统一的接口来处理这些数据，这样你就可以专注于模型的构建和训练，而不必担心数据加载和预处理的细节。
import datasets
from datasets import load_dataset


# ### 查看预训练数据集
# 加载了名为 "The Pile" 的数据集，由 EleutherAI 提供，也是常用于训练语言模型。
# split="train" 指定加载数据集的训练部分，
# streaming=True 表示以流式方式加载数据集，即数据会按需加载，不会一次性加载到内存中，适合处理大型数据集。
# https://huggingface.co/datasets/EleutherAI/pile
# **
# https://the-eye.eu/public/AI/pile/readme.txt  这个已经被删除了。
# pretrained_dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)

# "C4" (Common Crawl Corpus Cleaned) 数据集的英文版本。C4 是一个经过清理的大规模网络文本数据集，广泛用于训练和评估语言模型
# https://huggingface.co/datasets/legacy-datasets/c4
# pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)
pretrained_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

n = 5
print("预训练的数据集：")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(i)
print("-------------分割线-------------")
# ### 与您将使用的公司微调数据集相比
# JSON Lines（简称 JSONL）一种特殊的 JSON 格式，其中每一行包含一个完整的 JSON 对象，独立于其他行。这种格式特别适合处理大量数据，因为它允许逐行读取和解析，而无需一次性加载整个文件到内存中。
# 文件下载地址：https://github.com/FarshadAmiri/Fine-Tuning-on-LlaMa2/blob/main/lamini_docs.jsonl
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
print(instruction_dataset_df)
print("-------------分割线-------------")


# ### 格式化数据的各种方式

# 将 DataFrame 转换为字典格式
examples = instruction_dataset_df.to_dict()
text = examples["question"][0] + examples["answer"][0]
print(text)
print("-------------分割线-------------")

# 不同的数据进行拼接
if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]
print("打印不同的数据进行拼接：")
print(text)


prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""
question = examples["question"][0]
answer = examples["answer"][0]
text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)
print("使用提示词模版的数据：")
print(text_with_prompt_template)
# 输出类似格式：
# ### Question:
# What are the different types of documents available in the repository (e.g., installation guide, API documentation, developer's guide)?
#
# ### Answer:
# Lamini has documentation on Getting Started, Authentication, Question Answer Model, Python Library, Batching, Error Handling, Advanced topics, and class documentation on LLM Engine available at https://lamini-ai.github.io/.


# 定义提示模板:
prompt_template_q = """### Question:
{question}

### Answer:"""
# 确定样本数量: question 键下有多少个样本，这通常代表了整个数据集的大小。
num_examples = len(examples["question"])
# 初始化空列表，用于存储后续生成的数据。
# 存储只有文本的数据
finetuning_dataset_text_only = []
# 存储问题和答案的结构化数据。
finetuning_dataset_question_answer = []
# 遍历每个样本，对每个问题和答案进行格式化，并填充到之前定义的模板中。
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]
  # 包含问题和答案的完整文本
  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})
  # 这是只包含问题的文本（没有答案），这可能用于模型预测答案的场景。
  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})

print("finetuning_dataset_text_only 数据：")
pprint(finetuning_dataset_text_only[0])

print("finetuning_dataset_question_answer 数据：")
pprint(finetuning_dataset_question_answer[0])


# ### 常见的数据存储方式
with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)

# 加载数据集
finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
# 显示数据集的结构、包含的数据示例数、特征类型等信息。这有助于验证数据集是否正确加载，以及查看数据集的基本结构和内容。
print(finetuning_dataset)

