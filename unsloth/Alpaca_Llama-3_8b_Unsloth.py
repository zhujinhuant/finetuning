# -*- coding: utf-8 -*-

# Conda Installation
# Select either pytorch-cuda=11.8 for CUDA 11.8 or pytorch-cuda=12.1 for CUDA 12.1. If you have mamba, use mamba instead of conda for faster solving. See this Github issue for help on debugging Conda installs.
#
# conda create --name unsloth_env \
#     python=3.10 \
#     pytorch-cuda=<11.8/12.1> \
#     pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
#     -y
# conda activate unsloth_env
#
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#
# pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes



# 从unsloth库导入FastLanguageModel模块
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # 可以选择任何值！我们在内部自动支持RoPE缩放！ 模型一次可以接受多少个单词或标记（tokens）。
dtype = None # 自动检测数据类型。对于Tesla T4, V100使用Float16，对于Ampere+使用Bfloat16
load_in_4bit = True # 使用4位量化减少内存使用。也可以设置为False。

# 我们支持的4位预量化模型，下载速度提升4倍，不会出现内存溢出。
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # 新的Mistral v3模型，速度提升2倍！
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3模型，处理15万亿token，速度提升2倍！
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3模型，速度提升2倍！
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma模型，速度提升2.2倍！
] # 更多模型在 https://huggingface.co/unsloth

# 从预训练模型中加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # 可以选择任何模型，例如 teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # 如果使用受限制的模型如meta-llama/Llama-2-7b-hf，需使用token
)

"""我们现在添加LoRA适配器，这样我们只需要更新1%到10%的所有参数！增强原始模型 """
# 1. q_proj, k_proj, v_proj  注意力层
#    这三个模块分别对应于 Transformer 的注意力机制中的查询（Query）、键（Key）和值（Value）投影矩阵。在自注意力层中，输入序列被这三个矩阵转换以计算注意力分数：
#    q_proj (Query Projection)：将输入转换为查询向量，用于与键向量比较。
#    k_proj (Key Projection)：将输入转换为键向量，用于与查询向量比较生成注意力分数。
#    v_proj (Value Projection)：将输入转换为值向量，这些向量在计算得到注意力分数后将被加权求和，形成该层的输出。
# 2. o_proj  这是输出投影矩阵，用于将注意力机制的输出进一步转换为下一层或下一步处理的输入。
# 3. gate_proj, up_proj, down_proj 这些组件不是标准 Transformer 模型的一部分，它们可能是特定于实现的模块，用于控制信息流或实现特定的网络改进。具体来说：
#    gate_proj (Gate Projection)：可能用于实现门控机制，类似于 LSTM 或 GRU 中的门控，控制信息的传递和遗忘。
#    up_proj & down_proj (Up and Down Projections)：这些可能是用于特殊的参数化技巧或网络架构中的特定功能，如在LoRA中扩展和压缩信息流的线性变换。

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # 选择任何大于0的数！推荐的值包括8, 16, 32, 64, 128  r (Rank)：设定LoRA适配器中的低秩矩阵的秩。秩越低，改动的参数就越少，但可能会牺牲一些模型性能。
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],  # 指定要应用LoRA适配器的模型组件， 这些是注意力机制中的查询和键投影矩阵
    lora_alpha = 16,  # 控制LoRA适配器的学习率倍增器
    lora_dropout = 0, # 支持任何值，但0被优化使用 dropout率
    bias = "none",    # 支持任何值，但"none"被优化使用 ， 指定是否在LoRA适配器中使用偏置项。
    # [新功能] "unsloth"使用30%更少的VRAM，适合更大的批次大小
    use_gradient_checkpointing = "unsloth", # 真或"unsloth"用于非常长的上下文
    random_state = 3407,   #确保模型微调过程的可重复性。
    use_rslora = False,  # 我们支持排名稳定的LoRA
    loftq_config = None, # 以及LoftQ
)





"""
数据准备：
我们现在使用来自yahma的羊驼数据集，这是原始羊驼数据集中52K的过滤版本。您可以用自己的数据准备替换此代码段。
[注意]如果只想在完成句子上进行训练（忽略用户输入），请阅读TRL的文档[这里](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only)。
我们使用我们的`get_chat_template`函数获取正确的聊天模板。我们支持`zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old`以及我们自己优化的`unsloth`模板。
注意ShareGPT使用`{"from": "human", "value" : "Hi"}`而不是`{"role": "user", "content" : "Hi"}`，所以我们使用`mapping`来映射它。
对于文本完成，如小说写作，请尝试这个(https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)。
"""

from unsloth.chat_templates import get_chat_template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


"""
训练模型
现在让我们使用Huggingface TRL的“SFTTrainer”！
更多文档请点击此处：[TRL SFT文档](https://huggingface.co/docs/trl/sft_trainer). 
我们做了60个步骤来加快速度，但你可以为完整运行设置`num_train_epochs=1 `，并关闭`max_steps=None `。我们也支持TRL的DPOTrainer！
"""

# SFTTrainer：这是一个训练工具，用于对模型进行监督式微调。它是基于 Huggingface Transformers 的一个训练接口。
# TrainingArguments：这个类提供了许多配置选项，用于控制训练过程，包括批次大小、学习率、训练步骤数等。
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # 可以使训练对于短序列加速5倍。
    args = TrainingArguments(
        per_device_train_batch_size = 2,    #  每个设备的训练批次大小。
        gradient_accumulation_steps = 4,    # 梯度累积步骤数，用于在更新模型前积累更多的梯度，有助于稳定训练。
        warmup_steps = 5,                   # 预热步骤数，逐渐增加学习率以防模型一开始训练时更新太剧烈。
        max_steps = 60,                     # 最大训练步数，这里设置为60步。
        learning_rate = 2e-4,               # 学习率
        fp16 = not is_bfloat16_supported(), # 判断是否支持半精度和bfloat16精度，用于提高训练速度和降低内存消耗。
        bf16 = is_bfloat16_supported(),     # 判断是否支持半精度和bfloat16精度，用于提高训练速度和降低内存消耗。
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#显示当前内存统计信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# ********
# 开始模型训练的实际执行。
# ********
trainer_stats = trainer.train()

# 显示最终内存和时间统计数据
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")





"""
推论
让我们运行模型！由于我们使用的是“Llama-3”，因此使用“apply_chat_template”，并将“add_generation_prompt”设置为“True”进行推理。
"""

# 设置为推理模式，此模式特别优化了推理性能，使推理速度加快。
FastLanguageModel.for_inference(model) # 启用原生加速2倍的推理
# 准备一个包含斐波那契数列续写请求的消息列表，并通过模板处理这些消息，将其转换为模型可以理解的格式。
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")
# 使用 model.generate 方法进行文本生成。
# 此方法根据提供的输入（斐波那契数列的部分序列）生成接下来的数字序列。这里指定生成最多64个新令牌，并使用缓存加速生成过程。
outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
# 将生成的令牌序列解码成可读的文本。
tokenizer.batch_decode(outputs)

"""
您还可以使用TextStreamer进行连续推理，这样您就可以逐个查看生成令牌，而不是等待
"""
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)






"""
要将最终模型另存为LoRA适配器，请使用Huggingface的push_To_hub进行在线保存，或使用save-presetrained进行本地保存。
这只保存了LoRA适配器，而不是完整型号。要保存到16位或GGUF，请向下滚动！
"""
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")

"""
现在，如果你想加载我们刚刚保存用于推理的LoRA适配器，请将“False”设置为“True”：
"""
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

"""
您还可以使用Hugging Face的“AutoModelForPeftCausalLM”。
仅在未安装“unsloth”时使用此选项。它可能非常慢，因为不支持“4bit”模型下载，而Unloth的**推理速度要快2倍**。
"""
if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoModelForPeftCausalLM
    from transformers import AutoTokenizer
    model = AutoModelForPeftCausalLM.from_pretrained(
        "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")

"""
为VLLM保存到float16
我们还支持直接保存到`float16`。为float16选择“merged_16bit”，为int4选择“mergerd_4bit”。
我们还允许使用“lora”适配器作为后备方案。
使用`push_to_hub_merged`上传到您的Hugging Face帐户！你可以去https://huggingface.co/settings/tokens为您的个人key。
"""

# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

"""
GGUF/calla.cpp转换
为了保存到“GGUF”/“llama.cpp”，我们现在原生支持它！
我们克隆“llama.cpp”，并默认将其保存为“q8_0”。我们允许所有像`q4_k_m`这样的方法。
使用`savepretrained_gguf`进行本地保存，使用`push_to_hub_gguf'上传到HF。
一些支持的量化方法（完整列表见我们的[Wiki页面](https://github.com/unslothai/unsloth/wiki#gguf-量化选项）：
* `q8_0` - 快速转换。资源利用率高，但总体上可以接受。
* `q4_k_m` - 推荐。使用Q6_K作为注意力.wv和前馈.w2张量的一半，否则使用Q4_K。
* `q5_k_m` - 推荐。使用Q6_K作为注意力.wv和前馈.w2张量的一半，否则使用Q5_K。
"""

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

"""
现在，请在 `llama.cpp` 或像 `GPT4All` 这样的基于UI的系统中使用 `model-unsloth.gguf` 文件或 `model-unsloth-Q4_K_M.gguf` 文件。
你可以通过访问[这里](https://gpt4all.io/index.html)来安装 GPT4All。
我们已经完成了！如果你对Unsloth有任何疑问，我们有一个[Discord](https://discord.gg/u54VK8m8tk)频道！
如果你发现任何错误或想要获取最新的LLM信息，或需要帮助，加入项目等，欢迎加入我们的Discord！
其他一些链接：
1. Zephyr DPO加速2倍的[免费Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)
2. Llama 7b加速2倍的[免费Colab](https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing)
3. TinyLlama加速4倍，在1小时内完成全Alpaca 52K的[免费Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)
4. CodeLlama 34b在Colab上的A100加速2倍[免费Colab](https://colab.research.google.com/drive/1y7A0AxE3y8gdj4AVkl2aZX47Xu3P1wJT?usp=sharing)
5. Mistral 7b的[免费Kaggle版本](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook)
6. 我们还与🤗 HuggingFace合作发表了一篇[博客](https://huggingface.co/blog/unsloth-trl)，并且我们在TRL[文档](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth)中有介绍！
7. 像小说写作一样的文本完成[笔记本](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)
9. Gemma 6万亿令牌加速2.5倍的[免费Colab](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing)

<div class="align-center">
  <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/u54VK8m8tk"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://ko-fi.com/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Kofi button.png" width="145"></a></a> 如果可以，请支持我们的工作！谢谢！
</div>
"""