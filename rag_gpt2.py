import numpy as np
import pandas as pd
import re
import json
import nltk
#nltk.download('punkt')

from nltk.tokenize import sent_tokenize
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import faiss
import gradio as gr


device = 'cuda' if torch.cuda.is_available() else 'cpu'


save_dir = "trained_model/rag-embeddings"
file_path = "datasets/Plain-text-Wikipedia-(SimpleEnglish)/AllCombined.txt"

embd_txt = f"{save_dir}/embeddings.jsonl"
embd_npy = f"{save_dir}/embeddings.npy"


def clean_text(text):
    # Remove non-ASCII, control characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # non-ascii
    text = re.sub(r'\s+', ' ', text)             # normalize whitespace
    text = re.sub(r'[\*\[\]\=\_]', '', text)     # wiki symbols, if any remain
    text = re.sub(r'\.{3,}', '.', text)          # reduce long ellipses
    text = re.sub(r'\s([.,!?;:])', r'\1', text)  # remove space before punctuation
    text = re.sub(r'http\S+|www\S+', '', text)   # Remove hyperlinks
    return text.strip()


def read_raw_text(file_path):
    text = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = clean_text(raw)
            if line:
                text.append(line)
    return text


def create_sliding_chunks(text, max_tokens=300, stride=150):
    """
    Chunk the full cleaned text using a sliding window over sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for i, sentence in enumerate(sentences):
        tokens = sentence.split()
        sent_len = len(tokens)

        if current_len + sent_len <= max_tokens:
            current_chunk.append(sentence)
            current_len += sent_len
        else:
            # save the chunk
            chunks.append(' '.join(current_chunk))
            # slide window: keep last stride tokens
            backtrack = []
            backtrack_len = 0
            j = len(current_chunk) - 1
            while j >= 0 and backtrack_len < stride:
                back_tokens = current_chunk[j].split()
                backtrack.insert(0, current_chunk[j])
                backtrack_len += len(back_tokens)
                j -= 1
            current_chunk = backtrack + [sentence]
            current_len = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


#clean_text = clean_text(raw_text)
#chunks = create_sliding_chunks(clean_text, max_tokens=400, stride=200)

def load_emb(file_npy: str, file_jsonl: str):

    text_list = []

    with open(file_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text_list.append(obj["text"])

    # load binary
    embeddings = np.load(file_npy)

    assert len(embeddings) == len(text_list), "Embeddings and text count mismatch"

    return embeddings, text_list


model_emb = SentenceTransformer("data/all-MiniLM-L6-v2", device=device, local_files_only = True)

################################################################################################

if Path(embd_npy).exists() and Path(embd_txt).exists():

    embeddings, text = load_emb(embd_npy, embd_txt)

else:

    text = read_raw_text(file_path)

    embeddings = model_emb.encode(
        text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )

    os.makedirs(save_dir, exist_ok=True)

    np.save(embd_npy, embeddings)

    with open(embd_txt, "w", encoding="utf-8") as f:
        for t in text:
            obj = {"text": t}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Embeddings data saved to:", save_dir)

################################################################################################

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings) 

def retrieve_top_k(chunks, query:str, k:int=3):
    q_emb = model_emb.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)      # I = indices of top k
    return [chunks[i] for i in I[0]]

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
model_gpt2.eval()


def rag_generate_old(chunks:list,
                 user_input:str, 
                 k_retrieval:int=3, 
                 max_gen_len:int=100):
    # 1) retrieve related “memories”
    memories = retrieve_top_k(chunks, user_input, k=k_retrieval)

    # 2) build prompt: concatenate retrieved context + user query
    prefix = "\n---\n".join(memories)
    prompt = f"{prefix}\n\nUser: {user_input}\nAssistant:"

    max_gen_len = min(1024, inputs.input_ids.shape[1] + max_gen_len)

    # 3) tokenize & generate
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    outputs = model_gpt2.generate(
        **inputs,
        max_length=max_gen_len,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )

    # 4) decode completion (strip the prompt)
    gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return gen.strip()


def rag_generate(chunks: list,
                 user_input: str,
                 k_retrieval: int=3, 
                 max_gen_len: int=100,
                 temperature: float=0.7):

    print(f"[Debug] User input: {user_input}")

    # model’s absolute context window
    CONTEXT_WINDOW = tokenizer.model_max_length  # 1024 for GPT‑2

    # leave room for generation
    max_input_tokens = CONTEXT_WINDOW - (max_gen_len + 1)

    # retrieve related “memories”
    memories = retrieve_top_k(chunks, user_input, k=k_retrieval)

    # build prompt: concatenate retrieved context + user query
    def build_prompt(mems):
        prefix = "\n---\n".join(mems)
        return f"{prefix}\n\nUser: {user_input}\nAssistant:"

    # initial prompt with all retrieved memories
    prompt = build_prompt(memories)

    # if too long, drop memories until it fits
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    while input_ids.input_ids.shape[1] > max_input_tokens and memories:
        # drop the least relevant (last) item and rebuild prompt
        memories.pop(-1)
        prompt = build_prompt(memories)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    len_input = input_ids.input_ids.shape[1]

    # tokenize & generate
    outputs = model_gpt2.generate(
        **input_ids,
        max_new_tokens=max_gen_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=temperature
    )

    # decode completion
    answer = tokenizer.decode(outputs[0][len_input:], skip_special_tokens=True)
    answer = answer.split("\nAssistant:")[0].strip()
    return answer.split("\nUser:")[0].strip()


def add_to_memory(msg: str):
    emb = model_emb.encode([msg], convert_to_numpy=True)
    index.add(emb)


# Can you explain how transformers can work in your model?
#user_input = "Can you explain how transformers use attention mechanisms to understand context in a sentence?"
user_input = "Can you explain how transformers use attention mechanisms by LLM model to understand context?"
assistant_reply = rag_generate(text, user_input, 1, 100, 0.7)

print("\n\nAssistant:", assistant_reply)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()

    with gr.Row():
        msg = gr.Textbox(label="Your question", placeholder="Type here…")
        temp = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Generation temperature")
    with gr.Row():
        k_slider = gr.Slider(1, 5, value=1, step=1, label="k_retrieval")
        maxlen_slider = gr.Slider(50, 500, value=200, step=10, label="max_gen_len")
    
    def chat_fn(message, history, temperature, k_retrieval, max_gen_len):
        answer = rag_generate(
            text,
            user_input=message,
            k_retrieval=int(k_retrieval),
            max_gen_len=int(max_gen_len),
            temperature=float(temperature)
        )
        #history.append((message, answer))
        # gradio==6.9.0 + uses dict format for chatbot history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, history

    msg.submit(chat_fn, inputs=[msg, chatbot, temp, k_slider, maxlen_slider], outputs=[chatbot, chatbot])

demo.launch()
