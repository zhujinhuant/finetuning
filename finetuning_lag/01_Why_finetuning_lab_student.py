#!/usr/bin/env python
# coding: utf-8

# 比较微调和未微调模型
import os
import lamini

# lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")
lamini.api_key = "30970424127fc96621f3b9ae062314c101273ecf2f1c58dabede90dfb35a1865"
# 暂时没有数据，None
print("api_key:" + str(lamini.api_key))

# 导入模型运行库
from llama import BasicModelRunner

# #################################
# ### 尝试未微调的模型
# #################################

# 创建一个未微调的模型实例  (https://huggingface.co/meta-llama/Llama-2-7b-hf)
non_finetuned = BasicModelRunner("meta-llama/Llama-2-7b-hf")
# 使用未微调的模型生成对话
non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")
# 打印生成的对话
print("[Tell me how to train my dog to sit]")
print(non_finetuned_output)

# 使用未微调的模型回答其他问题
print("[What do you think of Mars?]")
print(non_finetuned("What do you think of Mars?"))
print("-"*30)

print("[taylor swift's best friend]")
print(non_finetuned("taylor swift's best friend"))
print("-"*30)

print("""[Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:]""")
print(non_finetuned("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))
print("-"*30)

# 比较微调模型
finetuned_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
# 使用微调模型生成对话
finetuned_output = finetuned_model("Tell me how to train my dog to sit")
print(finetuned_output)
print("-"*30)

# 使用微调模型和未微调模型分别回答特定格式的问题，比较差异
print(finetuned_model("[INST]Tell me how to train my dog to sit[/INST]"))
print("-"*30)
print(non_finetuned("[INST]Tell me how to train my dog to sit[/INST]"))
print("-"*30)
# 使用微调模型回答其他问题
print(finetuned_model("What do you think of Mars?"))
print("-"*30)
print(finetuned_model("taylor swift's best friend"))
print("-"*30)
print(finetuned_model("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))