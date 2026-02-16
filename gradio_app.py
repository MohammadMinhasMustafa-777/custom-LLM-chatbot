import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

base_model = "openai-community/gpt2"                                                # Base from HF
Hf_repo = "MohammadMinhasMustafa/GPT2_fine_tuned_for_custom_ChatBOT"                # HF repo

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, Hf_repo)                                   # Add adapters

def chatbot(prompt):
    inputs = tokenizer(prompt, return_tensors = 'pt')
    with torch.no_grad():                                                           # Saves Memory and Speed
        outputs = model.generate(
            inputs['input_ids'], 
            max_length = 150,
            temperature = 0.7,                                                      # Balanced creativity
            do_sample = True,                                                       # For variety
            top_k = 40,
            top_p = 5,                                                              # Focus on likely words
            repetition_penalty = 1.2                                                # Less repetition
            )
    generated_text = tokenizer.decode(outputs[0])
    return generated_text[len(prompt):].strip()  # Trim prompt, remove extras

gr.Interface(fn=chatbot, 
             inputs=gr.Textbox(label="Ask Anything", placeholder="Type your question here..."),       # Nicer input with label/placeholder
             outputs=gr.Textbox(label="Response"), 
             title="My Custom Chatbot", 
             description="Chat with AI tuned on my notes! Ask generic QnAs",        # Intro text"
             theme = gr.themes.Soft(primary_hue = "blue", secondary_hue="gray"),    # Attractive theme
             examples=[["What is your favorite color and why?"], 
             ["What is the capital city of Pakistan?"]],                            # Quick-click examples
             flagging_mode = "never",                                               # Hide flag button
             preload_example = False                                                # his prevents auto-filling the input on load.
             
             ).launch(share=True)
