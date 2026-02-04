import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

base_model = "distilgpt2"                                                       # Base from HF
Hf_repo = "MohammadMinhasMustafa/distilgpt2_fine_tuned_for_custom_ChatBOT"      # HF repo

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, Hf_repo)                               # Add adapters

def chatbot(prompt):
    inputs = tokenizer(prompt, return_tensors = 'pt')
    with torch.no_grad():               # Saves Memory and Speed
        outputs = model.generate(
            inputs['input_ids'], 
            max_length = 150,
            temperature = 0.7,          # Balanced creativity
            do_sample = True,           # For variety
            top_k = 40,
            top_p = 5,                  # Focus on likely words
            repetition_penalty = 1.2    # Less repetition
            )
    generated_text = tokenizer.decode(outputs[0])
    return generated_text[len(prompt):].strip()  # Trim prompt, remove extras

gr.Interface(fn=chatbot, inputs="text", outputs="text", title="My Custom Chatbot", description="Ask anything!").launch(share=True)
