from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_reply(user_input):
    prompt = f"Human: {user_input}\nAI:"

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 30,  # Kısa cevap için 30 token fazlası
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("AI:")[-1].strip()

iface = gr.Interface(
    fn=generate_reply,
    inputs=gr.Textbox(label="Mesajınızı Yazın"),
    outputs=gr.Textbox(label="GPT-2 Yanıtı"),
    title="GPT-2 Chatbot",
    description="Kısa ve öz cevaplar için optimize edildi."
)

iface.launch()
