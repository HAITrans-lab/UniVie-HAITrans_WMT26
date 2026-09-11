# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, set_seed
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig
import sys
#import evaluate
import numpy as np
from datasets import Dataset
import codecs
import json
import re 
import torch
import csv
from transformers import pipeline

from itertools import islice

import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="generate MT output Gemma3")

    parser.add_argument(
        "--json_file",
        type=str,
        default="combo_data/ELRC-2726-EMEA.en-pl.health.0-shot.test-hf.json",
        help="",
    )

    parser.add_argument(
        "--mt_output_file",
        type=str,
        default="combo_data/ELRC-2726-EMEA.en-pl.health.0-shot.gemma.test-mt.finetune.pl",
        help="",
    )


    parser.add_argument(
            "--model_id",
            type=str,
            default="models/gemma3-4b_merge-qlora_2epoch",
            help="[NOTE] use a merged model base+lora (merge_model.py)",
            )  


    parser.add_argument(
            "--size",
            type=int,
            default=10,
            help="batch size",
            )

    
    parser.add_argument(
            "--max_length",
            type=int,
            default=1024,
            help="",
        )

    parser.add_argument(
                "--top_p",
                type=float,
                default=0.9,
                help="",
            )
    
    
    args = parser.parse_args()

    return args

def contains_unicode_escape(s: str) -> bool:
    """Returns True if the string contains patterns like \\uXXXX or \\UXXXXXXXX."""
    # Matches \u followed by 4 hex digits OR \U followed by 8 hex digits
    pattern = r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}"
    return bool(re.search(pattern, s))

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def is_escaped_unicode(str):
    #how do I determine if this is escaped unicode?
    if is_ascii(str): # escaped unicode is ascii
        return True
    return False

def main():

    args = parse_args()
    #models/gemma3-12b-qlora_en-pl_fft_2epoch

    model_id = args.model_id #"models/gemma3-4b_merge-qlora_2epoch" #"models/gemma3-4b_merge-qlora_2epoch" #"models/gemma3-12b-merge-qlora_en-pl_fft_2epoch" #"  google/gemma-3-4b-it   
    json_file = args.json_file #"combo_data/ELRC-2726-EMEA.en-pl.health.0-shot.test-hf.json" #"combo_data/CCMatrix.es-eu.eng-tech.0-shot.test-hf.json" #"combo_data/CCMatrix.en-pl.eng-tech.5-shot.test-hf.json"
    mt_output_file = args.mt_output_file #"combo_data/ELRC-2726-EMEA.en-pl.health.0-shot.gemma.test-mt.finetune.pl" #"combo_data/CCMatrix.en-pl.eng-tech.5-shot.test-mt.finetune.csv" #"combo_data/CCMatrix.en-pl.eng-tech.5-shot.test-mt.csv"
    size = args.size #10
    max_length = args.max_lenght #512
    top_p = args.top_p #0.9

    print(json_file)
    print(mt_output_file)
    print(model_id)

    
   
    model = AutoModelForCausalLM.from_pretrained(model_id,  
                                                #attn_implementation="flash_attention_2",
                                                dtype='auto',
                                                device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()

    data_files = {}
    data_files["test"] = json_file
    data = load_dataset("json", data_files=data_files, encoding='utf-8')
     
    mt_output = open(mt_output_file, 'w', encoding='utf-8')
    #writer = csv.writer(mt_output, delimiter=',')
    
    pipe = pipeline("text-generation", 
                    model=model,                     
                    tokenizer=tokenizer,
                    device_map='auto', #auto for Qwen
                    batch_size=size)
    
    encoded_dataset = []
    
    
    for sample in data['test']:
        prompt = sample['translation']['prompt']     
        encoded_dataset.append([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
     
    prompt = pipe.tokenizer.apply_chat_template(encoded_dataset, 
                                                tokenize=False, 
                                                add_generation_prompt=True,
                                                )
    print('apply chat template')  
    #print(prompt)  
    outputs = pipe(prompt, 
                    max_new_tokens=max_length,
                    do_sample=True,
                    #temperature=0.6,
                    num_beams=1,
                    top_p=top_p,
                    top_k=0,
                    return_full_text=False
                    )

    print('mt')
    print(len(outputs))

    for output in outputs:
        #print(output)
        mt = output[0]["generated_text"]
      
        if contains_unicode_escape(mt):
            try:
                mt = mt.encode('utf-8').decode('unicode_escape')
                mt = re.sub(r'\n', ' ', mt, flags=re.DOTALL)
                mt = re.sub(r'^"', '', mt)
                mt = re.sub(r'"$', '', mt)
                #print(line)
            except:
                mt = re.sub(r'\n', ' ', mt, flags=re.DOTALL)
                mt = re.sub(r'^"', '', mt)
                mt = re.sub(r'"$', '', mt)
                #print(line)
        else:
           mt = re.sub(r'\n', ' ', mt, flags=re.DOTALL) 
       
        mt_output.write(f"{mt}\n")
       

    return

if __name__ == "__main__":
    main()