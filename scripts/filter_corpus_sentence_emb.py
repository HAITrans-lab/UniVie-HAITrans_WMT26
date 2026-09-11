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

    parser = argparse.ArgumentParser(description="filter top corpus")
    parser.add_argument(
        "--saved_file",
        type=str,
        default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.index.q.faiss",
        help="",
    )

    parser.add_argument(
        "--csv_file",
        type=str,
        default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.df.q.csv",
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
            default=1,
            help="",
            )

    
    parser.add_argument(
            "--query_file",
            type=str,
            default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.json",
            help="",
        )

    parser.add_argument(
                "--translation_json",
                type=str,
                default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.filter.json",
                help="",
            )
    
    
    args = parser.parse_args()

    return args




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
    
    saved_index = args.saved_index #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.index.30m.q.faiss" #"en-pl_ccmatrix.txt/CCMatrix.en-pl.clean.en.index.30m.q.faiss"
    csv_file = args.csv_file #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en.df.30m.q.csv" #"en-pl_ccmatrix.txt/CCMatrix.en-pl.clean.en.df.30m.q.csv"
    src_id = args.src_id #'en' #'en'
    trg_id = args.trg_id #'pl' #'pl'
    model_id = args.model_id #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top = args.top #1
    query_file =  args.query_file #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.test.json" #"es-eu_ccmatrix.txt/CCMatrix.es-eu.eng-tech.term-aya.test.json" #"en-pl_ccmatrix.txt/CCMatrix.en-pl.eng-tech.term-aya.test.json"
    translation_json = args.translation_json #f"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.term-aya.filter_{top}.test.json" #f"en-pl_ccmatrix.txt/CCMatrix.en-pl.eng-tech.term-aya.filter_{top}.test.json"
    print(query_file)
    print(translation_json)
      
    col = 'text'
    out_json = open(translation_json, 'w', encoding="utf-8")

    

    index = faiss.read_index(saved_index)

    data_files = {}
    data_files["train"] = query_file
    data = load_dataset("json", data_files=data_files, encoding='utf-8')
   
    embedder = SentenceTransformer(model_id, #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                   trust_remote_code=True,
                                    device="cuda:1")

    df = pd.read_csv(csv_file, index_col='id', encoding='utf-8')
    
    print('load files')
    #queries for index    

    for sample in data['train']:#['train']:
        #print(sample)
        term = sample['translation']['terms'] #TODO! terms
        synthetic = sample['translation'][src_id]
        
      
        emb_tmp = embedder.encode([synthetic])
        _, ID = index.search(emb_tmp, k=10)
        idx = ID.flatten().tolist()
        
        results = df.loc[idx, :]
       
        src_segments = results[col].to_list()
        trg_segments = results['trg'].to_list()
        examples = [[s,t] for s,t in zip(src_segments, trg_segments)][:top]
      
        src, trg = examples[0]
        out = {"translation": {'lang_pair': [src_id, trg_id], "terms": term,  'synthetic': synthetic, 'src': src, 'trg': trg} }
       
        x = json.dumps(out, indent=0, ensure_ascii=False)
        x = re.sub(r'\n', ' ', x, 0, re.M)
        out_json.write(x + "\n")

   
    return

if __name__ == '__main__':
     main()