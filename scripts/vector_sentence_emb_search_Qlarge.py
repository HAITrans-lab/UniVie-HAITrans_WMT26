# -*- coding: utf-8 -*-

import faiss
import numpy as np
import pandas as pd 
import pickle
import torch
#from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from sentence_transformers import SentenceTransformer
import gc
from datasets import load_dataset
import json
import re
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="vector emb search quantizied")


    parser.add_argument(
        "--query_file",
        type=str,
        default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.filter.json",
        help="",
    )

    parser.add_argument(
        "--saved_file",
        type=str,
        default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.index.q.faiss",
        help="",
    )

    parser.add_argument(
            "--src_id",
            type=str,
            default="en",
            help="",
            )

    parser.add_argument(
            "--trg_id",
            type=str,
            default="pl",
            help="",
            )

    parser.add_argument(
            "--model_id",
            type=str,
            default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            help="",
            )  


    parser.add_argument(
            "--top",
            type=int,
            default=0,
            help="",
            )

    
    parser.add_argument(
            "--domain",
            type=str,
            default="health",
            help="",
        )

    parser.add_argument(
                "--translation_json",
                type=str,
                default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.health.0-shot.json",
                help="",
            )
    
    
    args = parser.parse_args()

    return args


def prompt_str(domain, src_id, trg_id, src_segment, trg_segment, example_list):

    
    src = src_segment #sample['translation'][src_id]
    trg = trg_segment #sample['translation'][trg_id] 
    #few_shots = sample['translation']['few-shot-emb'][:few_size]
    
    
    few_shot_prompt = ""
    for few in example_list:
        src_f, trg_f = few
        # = s_t
        few_shot_prompt += f"{src_id}: {src_f}\n{trg_id}: {trg_f}\n"

    if few_shot_prompt:
        prompt = f"Translate the {src_id} source text to {trg_id} in the {domain} domain.\nRules:\nOutput strictly the {trg_id} translation.\nDo NOT repeat the {src_id} text.\nDo NOT include labels like \"{src_id}:\" or \"{trg_id}:\".\nDo NOT add quotation marks, explanations, or any extra text.\nExamples from the {domain} domain:\n{few_shot_prompt}Now translate the following:\n{src_id}: {src}\n{trg_id}: "
    else:
        prompt = f"Translate the {src_id} source text to {trg_id} in the {domain} domain.\nRules:\nOutput strictly the {trg_id} translation.\nDo NOT repeat the {src_id} text.\nDo NOT include labels like \"{src_id}:\" or \"{trg_id}:\".\nDo NOT add quotation marks, explanations, or any extra text.\nNow translate the following:\n{src_id}: {src}\n{trg_id}: "

    
    return prompt

def filter_n(results, query):
    filter = []
    for result in results:
        if result == query:
            continue
        else:
            filter.append(result)
    return filter

def index_result(csv_file, index_list):
    df = pd.read_csv(csv_file, index_col='id', encoding='utf-8')
    results = df.loc[index_list, :]
    return results


def main():

    args = parse_args()

    query_file = args.query_file #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.filter_1.test.json" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.eng-tech.term-aya.filter_1.test.json" #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.filter_1.test.json" #"en-pl_ccmatrix.txt/CCMatrix.en-pl.eng-tech.term-aya.filter_1.test.json" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.eng-tech.term-aya.filter_1.test.json"#
    saved_index = args.saved_index #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.index.30m.q.faiss" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.clean.es.index.30m.q.faiss" #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.index.30m.q.faiss"#"en-pl_ccmatrix.txt/CCMatrix.en-pl.clean.en.index.30m.q.faiss" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.clean.es.index.30m.q.faiss"#
    csv_file = args.csv_file #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.df.30m.q.csv" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.clean.es.df.30m.q.csv" #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.df.30m.q.csv"#"en-pl_ccmatrix.txt/CCMatrix.en-pl.clean.en.df.30m.q.csv" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.clean.es.df.30m.q.csv"#
    src_id = args.src_id #'en' #'en'
    trg_id = args.trg_id #'pl' #'pl'
    top = args.top #0
    domain = args.domain #"health"  #"health" # "engineering and technology"
    model_id = args.model_id
    translation_json = args.translation_json #f"combo_data/ELRC-2726-EMEA.en-pl.health.{top}-shot.test-hf.json" #f"en-pl_ccmatrix.txt/CCMatrix.en-pl.eng-tech.term-aya.{top}-shot.test.json" #f"es-eu_ccmatrix.txt/CCMatrix.es-eu.eng-tech.term-aya.{top}-shot.test.json" #
    col = 'text'
    out_json = open(translation_json, 'w', encoding="utf-8")

    print(query_file)
    print(translation_json)

  

    index = faiss.read_index(saved_index)

    data_files = {}
    data_files["train"] = query_file
    data = load_dataset("json", data_files=data_files, encoding='utf-8')
   

    embedder = SentenceTransformer(model_id, #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                               device="cuda:0")

    df = pd.read_csv(csv_file, index_col='id', encoding='utf-8')
    #df = None
    print('load files')
    #queries for index
    index_list = []

    lang_map = {'en':'English',
                'pl': 'Polish',
                'es': 'Spanish',
                'eu': 'Basque'}
    

    for sample in data['train']:#['train']:
        #print(sample)
        term = sample['translation']['terms'] #TODO! terms
        src = sample['translation'][src_id]
        if trg_id in sample['translation']:
            trg = sample['translation'][trg_id]
        else:
            trg = ""
        
        if top != 0:
            emb_tmp = embedder.encode([src])
            _, ID = index.search(emb_tmp, k=10)
            idx = ID.flatten().tolist()
            
            results = df.loc[idx, :]
            results = results[1:] #skip the first one distance=0
            #print(results)
            try:
                results = results[results[col].str.contains(src, regex=False) == False]
            except Exception as e:
                print("Error:", e)
           
            src_segments = results[col].to_list()
            trg_segments = results['trg'].to_list()
            examples = [[s,t] for s,t in zip(src_segments, trg_segments)][:top]
        else:
            examples = []
        #print(examples)
        prompt = prompt_str(domain, lang_map[src_id], lang_map[trg_id], src, trg, examples)
        out = {"translation": {"terms": term, 'prompt': prompt, 'examples': examples, 'lang_pair': [src_id, trg_id], 'src': src, 'trg': trg} }
        
        x = json.dumps(out, indent=0, ensure_ascii=False)
        x = re.sub(r'\n', ' ', x, 0, re.M)
        out_json.write(x + "\n")

   
    
    return

if __name__ == '__main__':
     main()