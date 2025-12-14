import random
import json
# Hugging Face的分词算法库
from tokenizers import ( 
    decoders, # 把分词后的token IDs还原成文本
    models, # 子词分词模型BPE, WordPiece... 对文本分割与映射成token IDs
    pre_tokenizers, # 文本预处理方式, 如按字节级别分割,空格分割等
    trainers, # 训练分词模型的工具
    Tokenizer # 分词器对象
)
import os

random.seed(42)

def train_tokenizer():
    # 读取JSON文件并提取文本数据
    def read_texts_from_jsonl(file_path, max_samples=100):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data = json.loads(line)
                yield data['text']
    
    data_path = "../dataset/pretrain_hq.jsonl"

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # 把文本转换为字节级别，不添加前缀空格

    # 定义特殊token
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] # 指示生成文本的终止、输入文本的开始和结束

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=6400, # 子词词汇表大小
        special_tokens=special_tokens, # 特殊tokens保证不会在BPE训练中被拆分
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 保存tokenizer
    tokenizer_dir = "../model/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/")

    # 手动创建配置文件
    config = {
        "add_bos_token": False, # 是否自动在文本开头添加 bos_token（如 `<
        "add_eos_token": False, # 是否自动在文本末尾添加 eos_token（如 `<
        "add_prefix_space": False, # Byte-level 分词时是否在文本前加空格。通常英文中启用（True）更好，中文中设为 False。
        # 特殊 token 的详细配置。包括 token 内容、是否为特殊 token、是否仅限单词等。key 是内部 token ID。
        "added_tokens_decoder": { 
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [], # 	除了 bos/eos/pad/unk 外，额外声明的特殊 token 列表。当前为空。
        "bos_token": "<|im_start|>", # 起始 token，通常用于语言模型的开头控制符，这里设为 `<
        "clean_up_tokenization_spaces": False, # 解码时是否清理 token 化带来的空格冗余。False 表示不清理。
        "eos_token": "<|im_end|>", # 结束 token，通常用于语言模型输出结束的标记，这里设为 `<
        "legacy": True, # 设置为 True 兼容旧版本 tokenizer 行为。推荐保持默认。
        "model_max_length": 32768, # 模型支持的最大 token 长度。超过将触发截断或报错。这里为 32768。
        "pad_token": "<|endoftext|>", # 用于对齐 padding 的特殊 token。此处为 `<
        "sp_model_kwargs": {}, # SentencePiece 模型的额外配置参数（当前为 BPE，未使用，故为空）。
        "spaces_between_special_tokens": False, # 是否在特殊 token 之间自动添加空格。设置为 False。
        "tokenizer_class": "PreTrainedTokenizerFast", # 指定 tokenizer 类型。Hugging Face 使用 "PreTrainedTokenizerFast" 支持 Rust 实现加速。
        "unk_token": "<|endoftext|>", # 用于标记未知词（out-of-vocabulary）的 token，这里也设为 `<
        # Jinja2 模板字符串，用于格式化对话数据为模型输入格式
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)
    
    print("Tokenizer training completed and saved.")

# 测试训练好的tokenizer
def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里?"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    # 将结构化的对话历史转换成一个一连串的字符串
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度
    actual_vocab_size = len(tokenizer)
    print("tokenizer实际词表长度: ", actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print("encoder长度：", len(model_inputs['input_ids']))

    input_ids = model_inputs["input_ids"]
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("decoder和原始文本是否一致：", response == new_prompt)

def main():
    train_tokenizer()
    eval_tokenizer()

if __name__ == "__main__":
    main()