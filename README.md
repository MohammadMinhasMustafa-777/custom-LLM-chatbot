## Custom LLM Chatbot

Mini-project: Fine-tuned GPT-2 on personal notes Q&A for a custom chatbot. Built with Gradio, deployed on Hugging Face Spaces.

# Features

Fine-tuning script using PEFT/LoRA.

Interactive Gradio app for chatting.

RAG for context-aware responses (optional).

# Setup

Git repo: https://github.com/MohammadMinhasMustafa-777/custom-LLM-chatbot.git

Install deps: pip install -r requirements.txt

Run fine-tuning (if needed): fine_tuned_LLM_for_custom_ChatBOT.ipynb (generates model).

Run app: python gradio_app.py

# Deployment

Hugging Face Space: https://huggingface.co/spaces/MohammadMinhasMustafa/custom_ChatBot

# Data

extracted_qna.json: Sample Q&A from notes.

# Notes

Model not included (large)—run fine_tuned_LLM_for_custom_ChatBOT.ipynb to generate.

For RAG, ensure sentence-transformers installed.