#!/usr/bin/env python
# coding: utf-8

# # 指令微调

# 导入必要的库
# 提供了一系列用于创建和操作迭代器的高效工具。迭代器用于帮助开发者生成数据流，尤其是在处理大数据或复杂循环结构时更有效率。
# 在这段代码中，itertools.islice 被用来从一个流式数据集中有效地获取前 m 个样本。这个函数非常适合用于大数据集或者流数据，因为它不需要加载整个数据集就可以访问数据的一个子集。这样可以节省内存和提高效率。
import itertools
# 一个用于处理存储在jsonlines格式中的数据的库。
# jsonlines格式是JSON的一种变体，每个JSON对象独立一行，这种格式适合大量数据的增量处理，因为它允许逐行读写，而无需一次性加载整个文件。
import jsonlines
# 加载数据集工具
from datasets import load_dataset
# 更好地显示数据的漂亮打印库
from pprint import pprint
# 导入用于运行模型和分词的类
from llama import BasicModelRunner
# AutoTokenizer: 用于自动加载与给定预训练模型匹配的分词器（Tokenizer）。分词器负责将文本转换为模型能理解的格式，即将字符串转换为数字表示（token ids）。
# AutoModelForCausalLM: 因果语言模型。用于加载能进行单向文本生成的模型，这种模型通常用于像文本续写这类的任务。这个类自动适配与指定的预训练模型对应的模型架构。
# AutoModelForSeq2SeqLM：序列到序列语言模型，用于加载那些设计来处理如机器翻译、文本摘要等任务的模型，这类任务通常需要模型将输入序列转换成新的输出序列。
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# 如果运行以上代码提示：None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
# 尽管 transformers 库已经安装，但是没有找到支持的深度学习框架（PyTorch、TensorFlow 2.0 或更高版本、Flax）。transformers 库依赖于这些框架之一来运行模型，虽然分词器和一些配置及文件处理工具仍然可以使用，但是模型相关的功能将不可用。
# pip install torch torchvision
# pip install tensorflow
# pip install flax


# ### 加载指令微调的数据集

# 加载专门为基于指令的微调准备的数据集
# AIpaca数据集地址：https://huggingface.com/datasets/tatsu-lab/alpaca
# 数据集名称：tatsu-lab/alpaca
instruction_tuned_dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
pprint(instruction_tuned_dataset)
# 显示指令微调数据集的前'm'个样本
m = 5
print("指令微调的数据集:")
top_m = list(itertools.islice(instruction_tuned_dataset, m))
for j in top_m:
  print(j)


# ### 两个提示模板

# 定义带有和不带有额外输入的提示生成模板
prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

# ### 填充提示（向提示中添加数据）
# 使用数据集中的数据填充模板，创建提示
processed_data = []
for j in top_m:
  if not j["input"]:
    processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])
  else:
    processed_prompt = prompt_template_with_input.format(instruction=j["instruction"], input=j["input"])
  processed_data.append({"input": processed_prompt, "output": j["output"]})

# 显示第一个处理过的提示以供验证
print("使用数据集中的数据填充模板，创建提示：")
pprint(processed_data[0])


# ### 将数据保存为jsonl文件

# 将处理过的数据保存到JSONL文件中，以便日后使用或共享
with jsonlines.open(f'alpaca_processed.jsonl', 'w') as writer:
    writer.write_all(processed_data)
print("将数据写入jsonl文件中。")


# ### 比较非指令微调与指令微调模型

# 加载标准数据集并显示
dataset_path_hf = "lamini/alpaca"
dataset_hf = load_dataset(dataset_path_hf)
print(dataset_hf)

import lamini
lamini.api_key = "30970424127fc96621f3b9ae062314c101273ecf2f1c58dabede90dfb35a1865"
# 加载并运行未经指令微调的模型
non_instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-hf")
non_instruct_output = non_instruct_model("Tell me how to train my dog to sit")
print("非指令微调输出（Llama 2 基础版）:", non_instruct_output)


# 加载并运行已经过指令微调的模型
instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
instruct_output = instruct_model("Tell me how to train my dog to sit")
print("指令微调输出（Llama 2）:", instruct_output)


### 尝试更小的模型

# 为更小的AI模型加载分词器和模型两个工具，分词器处理数据，model 运行模型。
# EleutherAI/pythia-70m 这是一个没有经过指令微调的7000万参数的模型。
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

# 定义一个推理函数（后续会有更深入的解释）
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # 对输入文本进行分词
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )
  # 使用模型生成输出
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )
  # 将生成的令牌解码为文本
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
  # 去掉提示从响应中获取答案
  generated_text_answer = generated_text_with_prompt[0][len(text):]
  return generated_text_answer

# 加载用于微调的数据集并显示
finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_path)
print("加载用于微调的数据集并显示，这里对于该问题Can Lamini generate technical documentation or user manuals for software projects? 的回答是比较精准的说明：")
print(finetuning_dataset)

# 对数据集中的测试样本进行推理
test_sample = finetuning_dataset["test"][0]
print("对数据集中的测试样本进行推理：")
print(test_sample)

print("非指令微调的推理数据，没有进行指令训练回答问题：Can Lamini generate technical documentation or user manuals for software projects? 就不太好有错误偏离主题：")
print(inference(test_sample["question"], model, tokenizer))


# ### 比较经过微调的小模型

# 加载经过指令处理微调的小模型
instruction_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
# 使用经过微调的小模型进行推理，并显示结果
print("指令微调的推理数据：进行指令训练回答问题：Can Lamini generate technical documentation or user manuals for software projects? 回答就很好。")
print(inference(test_sample["question"], instruction_model, tokenizer))
