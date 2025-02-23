import gradio as gr
from unsloth import FastLanguageModel
import torch

# 系统提示模板
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 加载模型
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/home/aeon/Desktop/GRPO-MEDreasoning/AEONA/myMedModel",  # 替换为你保存的模型路径
        max_seq_length = 2048,
        load_in_4bit = True,
        fast_inference = True,
        max_lora_rank = 64,
        gpu_memory_utilization = 0.5,
    )
    return model, tokenizer

# 生成回答
def generate_response(question):
    # 准备输入
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ], tokenize=False, add_generation_prompt=True)
    
    # 设置生成参数
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    
    # 生成回答
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
    )[0].outputs[0].text
    
    return output

# 创建Gradio界面
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# MedQwen3B-Reasoner 医疗推理助手")
        
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(
                    label="请输入您的医疗相关问题",
                    placeholder="例如:阿司匹林对心血管功能有什么影响?",
                    lines=3
                )
                submit_btn = gr.Button("提交问题")
            
            with gr.Column():
                answer = gr.Textbox(
                    label="AI回答",
                    lines=10,
                    interactive=False
                )
        
        submit_btn.click(
            fn=generate_response,
            inputs=question,
            outputs=answer
        )
        
    return demo

# 加载模型
print("正在加载模型...")
model, tokenizer = load_model()

# 启动UI
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )