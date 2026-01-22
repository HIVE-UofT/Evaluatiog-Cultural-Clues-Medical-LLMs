import pandas as pd
import numpy as np

# Tokenization utilities
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Runtime utilities
import math
import time
from tqdm import tqdm, trange
import json

login(token='')

system_prompt = """You are a medical expert assistant.
You are given multiple-choice medical questions.
Your task is to analyze the question and the provided answer options, apply correct medical reasoning, and choose the single best answer.

OUTPUT CONSTRAINTS (STRICT):
- The explanation must be exactly 2–3 sentences.
- Do NOT use bullet points, numbering, or headings.
- Do NOT include disclaimers, preambles, or meta-comments.
- Do NOT include extra whitespace or blank lines.
- The response must follow the exact format described below.

PENALTIES FOR VIOLATION:
- Any deviation from the required format, sentence count, or final answer line will be considered a critical error.
- Any text appearing after the final answer line will invalidate the response.
- Failure to end with the exact phrase will result in a zero score.

FINAL FORMAT (MANDATORY):
- After the explanation, output exactly one final line:
"The answer is X"
where X is one of (A, B, C, D, or E)."""


user_prompt = """Question:
{question}

Options:
{options}

Provide a medical explanation in exactly 2–3 sentences.
Strictly follow all formatting rules.
Then end the response with exactly this line and nothing else:
"The answer is X"
where X is the correct option letter."""

def form_message(question_text, options_dict):
  chat = {'conversations': []}
  chat['conversations'].append({
        "role": "system",
        "content": [
            {'type': 'text', 'text': system_prompt}
        ]
    })


  options_str = ''
  for k, v in options_dict.items():
      options_str += f'{k}. {v}\n'

  chat['conversations'].append({
        "role": "user",
        "content": [
            {'type': 'text', 'text': user_prompt.format(question=question_text,
                               options=options_str)}
        ]
    })
    
  return chat

with open('/home/haji80as/EMBC_project/MedQA/final_augment_test_questions_v2.json', 'r') as f:
    dataset = json.loads(f.read())
    f.close()


all_chats = []
for d in dataset:
    chats = {}
    chats['original'] = form_message(d['original_question'], d['options'])['conversations']
    chats['neutral'] = form_message(d['neutral_question'], d['options'])['conversations']
    for t in ['t1', 't2', 't3']:
        chat_obj = {}
        for e in ['i', 'c', 'ic']:
            chat_obj[e] = form_message(d[t][e], d['options'])['conversations']
        chats[t] = chat_obj.copy()
            
    all_chats.append(chats)

pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it")


all_answers = []
for chats in tqdm(all_chats):
    ans_obj = {}
    resp = pipe(chats['original'], max_new_tokens=4096)
    ans_obj['original'] = resp[0]['generated_text'][2]['content']
    resp = pipe(chats['neutral'], max_new_tokens=4096)
    ans_obj['neutral'] = resp[0]['generated_text'][2]['content']
    for t in ['t1', 't2', 't3']:
        ans_obj[t] = {}
        for e in ['i', 'c', 'ic']:
            resp = pipe(chats[t][e], max_new_tokens=4096)
            ans_obj[t][e] = resp[0]['generated_text'][2]['content']
    all_answers.append(ans_obj)
    
    if len(all_answers) % 10 == 0:
        with open('/home/haji80as/EMBC_project/new_answers/MedGemma4B_short_augment_answers.json', 'w') as f:
            f.write(json.dumps(all_answers))
            f.close()

with open('/home/haji80as/EMBC_project/new_answers/Final_MedGemma4B_short_augment_answers.json', 'w') as f:
            f.write(json.dumps(all_answers))
            f.close()
