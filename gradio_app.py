import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

fine_tuned_model_path = r"C:\Python 2k25\4_LLMs_Basics\Day_5_Mini_Project_Custom_LLM_Chatbot\Model_for_GPT-2_fine_tuned_for_custom_ChatBOT"

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)

def chatbot(prompt):
    inputs = tokenizer(prompt, return_tensors = 'pt')
    with torch.no_grad():           # Saves Memory and Speed
        outputs = model.generate(
            inputs['input_ids'], 
            max_length = 50,
            temperature = 0.7,          # Balanced creativity
            do_sample = True,           # For variety
            top_k = 40,
            top_p = 5,                  # Focus on likely words
            repetition_penalty = 1.2    # Less repetition
            )
    generated_text = tokenizer.decode(outputs[0])
    return generated_text[len(prompt):].strip()  # Trim prompt, remove extras

gr.Interface(fn=chatbot, inputs="text", outputs="text").launch(share=True)
