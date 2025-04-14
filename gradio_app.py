import gradio as gr
from transformers import pipeline

# Load smaller model for faster performance
chatbot = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1  # -1 for CPU, 0 for GPU
)

def respond(message, history):
    response = chatbot(
        message,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=50256,  # GPT2 pad token
        do_sample=True,
        temperature=0.7
    )
    return response[0]['generated_text']

gr.ChatInterface(
    respond,
    title="Local AI Chat",
    description="A completely local chatbot"
).launch()