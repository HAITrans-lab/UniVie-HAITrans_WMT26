# -*- coding: utf-8 -*-
import re
import json
import re
import torch
from transformers import pipeline, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="synthetic sentence LLM")
    parser.add_argument(
        "--term_file",
        type=str,
        default="IATE_eng/IATE_eng-tech.en",
        help="",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.json",
        help="",
    )

    parser.add_argument(
            "--src_id",
            type=str,
            default="en",
            help="",
        )

    parser.add_argument(
                "--source",
                type=str,
                default="English",
                help="",
            )

    parser.add_argument(
                "--domain",
                type=str,
                default="health",
                help="",
            )

    parser.add_argument(
                "--model_id",
                type=str,
                default="CohereLabs/aya-expanse-32b",
                help="",
            )  
    
    args = parser.parse_args()

    return args

    
def load_file(file_lines):
    lines = []
    for line in file_lines:
        line = line.strip()
        lines.append(line)
    return lines



def load_term(term_file):
    term_lines = open(term_file, "r", "utf-8")
    term_list = []
    for line in term_lines:
        line = line.strip()
        term_list.append(line)
    return list(set(term_list))

def msg(source, domain, terms):
    messages = []
    for term in terms:
        messages.append([{"role": "user", "content": f"Produce one sentence in {source} using the following term from the {domain} domain: \"{term}\""}])
    return messages


def main():

    args = parse_args()
    
    term_file = args.term_file #"IATE_med/IATE_export.en" #"IATE_eng/IATE_eng-tech.en"
    out_file =  args.out_file #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.json" #"en-pl_ccmatrix.txt/CCMatrix.en-pl.eng-tech.term-aya.json" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.eng-tech.term-aya.json" #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term.json"
    src_id = args.src_id #'en'
    source = args.source #"English"
    domain = args.domain #"health"
    model_id = args.model_id #"CohereLabs/aya-expanse-32b"
    out_json = open(out_file, 'w', encoding="utf-8")
   
    terms  = load_term(term_file)
    
    
    i = 0
    j = 0
    
    print("#terms:", len(terms))
      
   
    bnb_config =  BitsAndBytesConfig(load_in_8bit=True) 

    pipe = pipeline("text-generation", 
                    model=model_id, 
                    model_kwargs={"quantization_config":bnb_config, 
                                  'dtype':torch.bfloat16},
                                  #'attn_implementation':"flash_attention_2"},
                    device_map="auto")
    

    start = 0
    end = len(terms)
    
    step = 40

    #for term in terms:
    for i in range(start, end, step):
        x = i
        t = terms[x:x+step]
        messages = msg(source, domain, t)
        prompt = pipe.tokenizer.apply_chat_template(messages,
                                                    tokenize=False, 
                                                    #enable_thinking=False,
                                                    add_generation_prompt=True)
        #print(prompt)
        outputs = pipe(prompt, 
                       max_new_tokens=256,  
                       do_sample=True,
                       temperature=0.6,
                       top_p=0.95,
                       return_full_text=False)
        #print(outputs)
        for j, output in enumerate(outputs):
            text = output[0]["generated_text"]
           
            out = {"translation": {"term": t[j], src_id: text} }
            #print(out)
            
            x = json.dumps(out, indent=0, ensure_ascii=False)
            x = re.sub(r'\n', ' ', x, 0, re.M)
            out_json.write(x + "\n")
    
     
    return

if __name__ == '__main__':
     main()
    
