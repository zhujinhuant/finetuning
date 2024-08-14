#!/usr/bin/env python
# coding: utf-8

# pandas用于数据分析
import pandas as pd
# datasets来自huggingface，常用于加载和管理大型数据集。
import datasets

from pprint import pprint
# AutoTokenizer用于自动加载预训练的分词器。
from transformers import AutoTokenizer


# ## 文本分词

# 加载了一个名为"EleutherAI/pythia-70m"的"预训练分词器"
# 分词器是用于将文本转换为模型可以理解的数值形式（即token化）。
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

text = "Hi, how are you?"
# input_ids是token的数值表示
encoded_text = tokenizer(text)["input_ids"]
# [12764, 13, 849, 403, 368, 32]
print(text+"的token数值表示：", encoded_text)
# 将token化的文本解码回原始文本，并打印。
decoded_text = tokenizer.decode(encoded_text)
print("将token化的文本解码回原始文本: ", decoded_text )


# ## 同时对多个文本进行分词


list_texts = ["Hi, how are you?", "I'm good", "Yes"]
encoded_texts = tokenizer(list_texts)
# [[12764, 13, 849, 403, 368, 32], [42, 1353, 1175], [4374]]
print("对多个文本进行编码: ", encoded_texts["input_ids"])


# ## 使用填充和截断

# 在操作的是固定大小的张量（Tensors），所以文本编码的长度应当是相同的。为了解决这一问题，我们经常使用一种名为"填充"pandding的策略。
# 填充是一种处理不同长度文本编码的方法。对于填充词元，你需要指定用哪个数字来表示填充。（tokenizer.pad_token = tokenizer.eos_token）
tokenizer.pad_token = tokenizer.eos_token
# padding = True 表示填充，Yes，后面就有很多0  [4374, 0, 0, 0, 0, 0]
encoded_texts_longest = tokenizer(list_texts, padding=True)
print("使用填充: ", encoded_texts_longest["input_ids"])

# 有些不需要很多文本的内容，可以设定截断
# 截断，设置最大长度为3，并对超过这个长度的输入进行截断（truncation）。
encoded_texts_truncation = tokenizer(list_texts, max_length=3, truncation=True)
print("使用截断: ", encoded_texts_truncation["input_ids"])

# 设置截断发生在文本的左侧。
tokenizer.truncation_side = "left"
encoded_texts_truncation_left = tokenizer(list_texts, max_length=3, truncation=True)
print("截断左侧文本: ", encoded_texts_truncation_left["input_ids"])

encoded_texts_both = tokenizer(list_texts, max_length=3, truncation=True, padding=True)
print("同时使用截断和填充: ", encoded_texts_both["input_ids"])




# ## 准备指令数据集

# jsonl 数据集名称，在前几课程中已经生成
filename = "lamini_docs.jsonl"
# 读取本地指令数据集，lines=True参数指明每一行都是一个独立的JSON对象。
instruction_dataset_df = pd.read_json(filename, lines=True)
examples = instruction_dataset_df.to_dict()
# 构建文本数据，检查不同的键来决定如何组合文本，这样处理的目的是使模型能够理解和生成相应的输出。
if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]

# 创建微调数据集：格式化问题和答案的表示方式。
# 添加到提示词模版组合成提示词
prompt_template = """### Question:
{question}

### Answer:"""

num_examples = len(examples["question"])
finetuning_dataset = []
# 遍历所有的问题和答案，将它们格式化并存入finetuning_dataset列表中。这个列表最终包含了所有用于模型训练的数据点，每个数据点都是一个字典，包含格式化后的问题和答案。
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]
  text_with_prompt_template = prompt_template.format(question=question)
  finetuning_dataset.append({"question": text_with_prompt_template, "answer": answer})

print("微调数据集中的一个数据点:")
# {'question': "### Question:\nWhat are the different types of documents available in the repository (e.g., installation guide, API documentation, developer's guide)?\n\n### Answer:",
# 'answer': 'Lamini has documentation on Getting Started, Authentication, Question Answer Model, Python Library, Batching, Error Handling, Advanced topics, and class documentation on LLM Engine available at https://lamini-ai.github.io/.'}
pprint(finetuning_dataset[0])


# ## 分词单个示例

# 对单个示例进行分词，并设置返回的tensor类型为NumPy数组，并添加必要的填充。
text = finetuning_dataset[0]["question"] + finetuning_dataset[0]["answer"]
tokenized_inputs = tokenizer(text,return_tensors="np",padding=True)
print("token数值表示:",tokenized_inputs["input_ids"])

# 设置最大长度和截断
max_length = 2048 #大长度2048
# 根据实际的长度和2028比大小，取最小的，避免在处理较短的文本时不必要的截断
max_length = min(tokenized_inputs["input_ids"].shape[1],max_length,)

# 使用分词器tokenizer处理文本text，设置返回类型为NumPy数组，并启用截断功能，使用上面计算得到的max_length作为最大长度
tokenized_inputs = tokenizer(text,return_tensors="np",truncation=True,max_length=max_length)
print("token数值，2048表示:",tokenized_inputs["input_ids"])


# ## 对指令数据集进行标记(和上面的测试代码类似，只是封装成一个函数)

# 为数据集中的每个条目调用的。它将文本分词，设置填充和截断，然后返回处理后的数据。
def tokenize_function(examples):
  # 根据不同的字段组合文本
  if "question" in examples and "answer" in examples:
    text = examples["question"][0] + examples["answer"][0]
  elif "input" in examples and "output" in examples:
    text = examples["input"][0] + examples["output"][0]
  else:
    text = examples["text"][0]

  # 设置填充符为结束符，进行分词处理
  tokenizer.pad_token = tokenizer.eos_token
  tokenized_inputs = tokenizer(text,return_tensors="np",padding=True,)

  # 设置最大长度，并执行截断处理
  max_length = min(tokenized_inputs["input_ids"].shape[1],2048)
  tokenizer.truncation_side = "left"
  tokenized_inputs = tokenizer(text,return_tensors="np",truncation=True,max_length=max_length)
  return tokenized_inputs

# 加载和处理数据集,tokenize_function进行分词处理，并输出处理后的数据集。
finetuning_dataset_loaded = datasets.load_dataset("json", data_files=filename, split="train")
# 将分词函数应用到数据集的每一个数据项。drop_last_batch 是我们为了处理混合大小的输入而做的选择
tokenized_dataset = finetuning_dataset_loaded.map(tokenize_function,batched=True,batch_size=1,drop_last_batch=True)
print("加载和处理数据集：",tokenized_dataset)
print("加载输出第一个数据集：", tokenized_dataset[0])
# 添加标签列，我们必须要添加一个labels列，以便HuggingFace处理
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])
print("添加标签列：", tokenized_dataset[0])
# 数据集划分，train_test_split，将测试集大小指定为整个数据集的10%，也可以改成其他比例，shuffle随机打乱数据集的顺序
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
print(" 数据集划分: ", split_dataset)


# ## 加载更多数据集

finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = datasets.load_dataset(finetuning_dataset_path)
print(finetuning_dataset)
# 包括Taylor Swift和BTS相关的数据集
taylor_swift_dataset = "lamini/taylor_swift"
bts_dataset = "lamini/bts"
open_llms = "lamini/open_llms"

dataset_swiftie = datasets.load_dataset(taylor_swift_dataset)
print(dataset_swiftie["train"][1])


# 将处理后的数据集上传到Huggingface Hub，需要先安装huggingface_hub库，登录Huggingface账号后，使用push_to_hub方法上传数据集。
# !pip install huggingface_hub
# !huggingface-cli login
# split_dataset.push_to_hub(dataset_path_hf)




