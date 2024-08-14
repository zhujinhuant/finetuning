#!/usr/bin/env python
# coding: utf-8

# # 训练

# ## 在GPU上运行的代码其实只需要几行（例如在Lamini上）。
# 从llama库导入BasicModelRunner类
# from llama import BasicModelRunner

# 实例化一个模型运行器，指定模型名称
# model = BasicModelRunner("EleutherAI/pythia-410m")
# 从jsonlines文件加载数据，指定输入和输出的键
# model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
# 开始训练模型，并设置为公开模型
# model.train(is_public=True)
# 1. 选择基础模型。
# 2. 加载数据。
# 3. 训练模型。返回模型ID、控制台和交互界面。

# ### 让我们看看支持这些功能的核心代码！这是Lamini的`llama`库的开源核心 :)


# 导入所需的库
import os
import lamini

# 设置API URL和密钥环境变量
# lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
# lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")
lamini.api_key = "30970424127fc96621f3b9ae062314c101273ecf2f1c58dabede90dfb35a1865"


# 导入更多库
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines

# 导入工具函数和transformers库中的类
from utilities import *
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner

# 创建日志记录器
logger = logging.getLogger(__name__)
global_config = None

# ### 加载Lamini文档数据集

# 设置数据集名称和路径，导入微调数据。也可以使用HuggingFace导入数据。
dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
# 通过标识，确实是否使用HuggingFace
use_hf = False

# 修改数据集路径和使用方式
dataset_path = "lamini/lamini_docs"
use_hf = True

# ### 设置模型、训练配置和分词器

# 设置模型名称，这个model只有7000万参数
model_name = "EleutherAI/pythia-70m"

# 创建训练配置配置，方便统一管理
training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

# 初始化分词器，并设置填充令牌
# 加载了一个名为"EleutherAI/pythia-70m"的"预训练分词器"
# 分词器是用于将文本转换为模型可以理解的数值形式（即token化）。
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

# 打印训练和测试数据集
print(train_dataset)
print(test_dataset)

# ### 加载模型

# 从预训练加载模型
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 检测GPU设备数量并选择设备（PyTorch代码，计算你有多少个CUDA设备）
device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    # 确保代码能够适应 MPS（ Metal Performance Shaders）设备
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")



# 将模型运行在选定的cpu或者gpu上
model_info = base_model.to(device)
print("model的信息：",model_info)

# ### 定义执行"推理"的函数
# max_input_tokens  输入到模型中的词元数量
# max_output_tokens  预期输出的词元数量
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # 首先：分词（对输入的文本进行分词）
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )
  # 生成文本（分词好传入模型，这些词元token，必须要放在同一个设备上）
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    # 默认100，生成内容越多，需要的时间会更长
    max_length=max_output_tokens
  )

  # 解码生成的文本（根据词元解码它）
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # 去除初始化的提示词部分（因为，他同时输出了提示词和生成的内容）
  generated_text_answer = generated_text_with_prompt[0][len(text):]
  # 返回生成的文本答案
  return generated_text_answer

# ### 尝试基础模型（第一个测试集问题）
# 这里回答可能会比较奇怪，也可能没有真正回答问题。这也是后面训练的目的。
test_text = test_dataset[0]['question']
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))


###########################
# ### 设置训练
###########################
# 训练的最大步数（一个步数就是一个训练数据批次，如果你的批次大小是2000，那就是2000个数据点）
max_steps = 3
# 训练模型的名称（数据集名称+最大步数）
trained_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = trained_model_name

# 创建训练参数
training_args = TrainingArguments(
  learning_rate=1.0e-5,  # 学习率
  num_train_epochs=1,  # 训练轮数
  max_steps=max_steps,  # 最大步数，覆盖num_train_epochs
  per_device_train_batch_size=1,  # 训练批次大小
  output_dir=output_dir,  # 模型保存路径
  overwrite_output_dir=False,  # 是否覆盖输出目录
  disable_tqdm=False,  # 是否禁用进度条
  eval_steps=120,  # 每多少步进行一次评估
  save_steps=120,  # 每多少步保存一次模型
  warmup_steps=2,  # 预热步数
  per_device_eval_batch_size=1,  # 评估批次大小
  evaluation_strategy="steps",
  logging_strategy="steps",
  logging_steps=1,
  optim="adafactor",
  gradient_accumulation_steps = 4,  # 梯度累积步数
  gradient_checkpointing=False,  # 是否开启梯度检查点
  load_best_model_at_end=True,  # 训练结束时是否加载最佳模型
  save_total_limit=1,  # 最多保存几个模型
  metric_for_best_model="eval_loss",  # 选择最佳模型的指标
  greater_is_better=False  # 指标是越大越好还是越小越好
)

# 计算模型的浮点运算次数，以了解此基础模型的内存占用
model_flops = (
  base_model.floating_point_ops(
    {
       "input_ids": torch.zeros(
           (1, training_config["model"]["max_length"])
      )
    }
  )
  * training_args.gradient_accumulation_steps
)
# 打印出基础信息，包括内存和flops
# （“Flops”代表“每秒浮点运算次数”是衡量计算机性能的一个指标，特别是它在一秒钟内能执行多少涉及浮点数的计算。
# 当提到模型的Flops时（如人工智能模型），它通常指模型完成一次推理或预测周期所需执行的浮点运算总数。这常用于估计模型的计算负载或效率。
# 例如，如果一个人工智能模型在进行一次预测时需要执行数十亿次浮点运算，那么它的Flops就会相应很高，这表明该模型在执行时对计算资源的需求较大。
print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

# 创建训练器，加载模型，各种训练参数，数据集等 主训练设计的一个类
trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
##################
# ### 开始训练，最重要的一步。
##################
training_output = trainer.train()

# ### 本地保存模型
save_dir = f'{output_dir}/final'
trainer.save_model(save_dir)
print("Saved model to:", save_dir)


# 加载 微调过的模型，local_files_only=True，确保不会从HuggingFace hub的服务器下载
finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)

# 将模型移动到设备
finetuned_slightly_model.to(device)



# ### 运行微调过的模型（可能没有变的更好，因为之训练了几次）
test_question = test_dataset[0]['question']
print("Question input (test):", test_question)
print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer))

# ### 输出测试答案
test_answer = test_dataset[0]['answer']
print("Target answer output (test):", test_answer)


# ### 运行训练更久的模型（可能需要一些时间，可能是30分钟）
finetuned_longer_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")

finetuned_longer_model.to(device)
print("Finetuned longer model's answer: ")
print(inference(test_question, finetuned_longer_model, tokenizer))

# ### 运行更大的训练模型并探索内容审查
bigger_finetuned_model = BasicModelRunner(model_name_to_id["bigger_model_name"])
bigger_finetuned_output = bigger_finetuned_model(test_question)
print("Bigger (2.8B) finetuned model (test): ", bigger_finetuned_output)


# 主题纠偏：就是鼓励模型不要偏离主题太远。（例如在一些训练数据中：让我们的讨论限制在Lamini相关的话题上。）

# ### 探索小模型的内容审查
# 首先，尝试未经微调的基础模型：
base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
print(inference("What do you think of Mars?", base_model, base_tokenizer))

# ### 尝试经过微调的小模型进行内容审查（别且做个主题纠偏）
print(inference("What do you think of Mars?", finetuned_longer_model, tokenizer))



# ### 使用Lamini在3行代码内微调模型（可以托管到服务器上进行训练，可以到对应的网站查看:https://app.lamini.ai/account,等于并注册）
model = BasicModelRunner("EleutherAI/pythia-410m")
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True)  # 公开训练模型

# # 评估模型
# out = model.evaluate()
#
# # 处理评估结果，生成数据表
# lofd = []
# for e in out['eval_results']:
#     q  = f"{e['input']}"
#     at = f"{e['outputs'][0]['output']}"
#     ab = f"{e['outputs'][1]['output']}"
#     di = {'question': q, 'trained model': at, 'Base Model' : ab}
#     lofd.append(di)
# df = pd.DataFrame.from_dict(lofd)
# style_df = df.style.set_properties(**{'text-align': 'left'})
# style_df = style_df.set_properties(**{"vertical-align": "text-top"})
# style_df
#
# # 这段代码主要展示了如何使用Python和相关库来训练和使用深度学习模型进行自然语言处理。
