# -*- coding: utf-8 -*-

# Reference: kstathou/acl-search-engine

# !pip install faiss-cpu --no-cache
# !pip install sentence_transformers

import faiss
import numpy as np
import pandas as pd 
import pickle
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np 
import re
import gc
from random import shuffle
from more_itertools import sliced
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="vector emb index quantizied")
    parser.add_argument(
        "--src_file",
        type=str,
        default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en",
        help="",
    )

    parser.add_argument(
        "--trg_file",
        type=str,
        default="en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.pl",
        help="",
    )

    parser.add_argument(
                "--model_id",
                type=str,
                default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                help="",
            )  

    parser.add_argument(
                    "--max_size",
                    type=int,
                    default=30000000,
                    help="",
                )

    parser.add_argument(
                        "--max_train",
                        type=int,
                        default=4000000,
                        help="",
                    )

    parser.add_argument(
                        "--emb_size",
                        type=int,
                        default=384,
                        help="",
                        )

    parser.add_argument(
                        "--nlist",
                        type=int,
                        default=1000,
                        help="",
                            )

    parser.add_argument(
                        "--m",
                        type=int,
                        default=16,
                        help="",
                        )

    parser.add_argument(
                        "--nbits",
                        type=int,
                        default=8,
                        help="",
                            )
    
    args = parser.parse_args()

    return args

embedder = None


def load_files(src_lines, trg_lines):
    lines = []
    i = 0
    for s, t in zip(src_lines, trg_lines):
        s = s.strip()
        t = t.strip() 
        s = re.sub(r'\n+', ' ', s, 0, re.M)
        t = re.sub(r'\n+', ' ', t, 0, re.M)
        lines.append([s, t])
        i += 1

    return lines

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def train_corpus(df_file, col="text", out_file="faiss_index.faiss", max_size=2000000, embedder=None, emb_size=384, nlist=1000, m=16, nbits=8):
    
    quantizer = faiss.IndexFlatL2(emb_size)
    index = faiss.IndexIVFPQ(quantizer, emb_size, nlist, m, nbits) #30000, 64, 8

    chunk_size = 50000
    tmp_data = []
    i = 0
    for chunk in pd.read_csv(df_file, index_col='id', chunksize=chunk_size):
       
        try:
            emb_tmp = embedder.encode([str(element) for element in chunk[col].to_list()], #chunk[col].to_list(),
                                    batch_size=50)
        except Exception as e:
            print("Error:", e)
            #print(chunk[col].to_list())
            continue
      
        tmp_data.extend(emb_tmp)
        #index.add_embeddings(emb_tmp, index_val)
        
        del(chunk)
        del(emb_tmp)

        if i >= max_size:
            break
        gc.collect()
        i += chunk_size
    tmp_data = np.array(tmp_data)
    index.train(tmp_data)
    print('train index', max_size)
    del(tmp_data)
   
    return index

def index_corpus(index, df_file, col="text", out_file="faiss_index.faiss", source_lang="eng_Latn", embedder=None):
   
       
    gc.collect() 
   
    chunk_size = 50000

    #index_slices = sliced(range(len(df)), chunk_size)
    for chunk in pd.read_csv(df_file, index_col='id', chunksize=chunk_size):
       
        print(chunk)
      
        try:
            emb_tmp = embedder.encode([str(element) for element in chunk[col].to_list()], #chunk[col].to_list(),
                                    batch_size=50)
        except Exception as e:
            print("Error:", e)
            #print(chunk[col].to_list())
            continue
        #emb_tmp = emb_tmp.detach().cpu().numpy()
        index_val = chunk.index.values
      
        index.add_with_ids(emb_tmp, index_val)
        #index.add_embeddings(emb_tmp, index_val)
        del(chunk)
        del(emb_tmp)
        del(index_val)
        gc.collect()
     
   
    faiss.write_index(index, out_file)
    
    return index

def main():

    args = parse_args()

    src_file = args.src_file #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.en-filtered.en"
    trg_file = args.trg_file #"en-pl_emea.txt/ELRC-2726-EMEA.en-pl.pl-filtered.pl"#"en-pl_ccmatrix.txt/CCMatrix.en-pl.clean.pl" 
    model_id = args.model_id #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_size = args.max_size # 30000000
    max_train = args.max_train #4000000
    emb_size = args.emb_size #384
    nlist = args.nlist # 1000
    m = args.m # 16
    nbits = args.nbits # 8
    out_file = f"{src_file}.index.q.faiss"
    df_file = f"{src_file}.df.q.csv"
    src_lang = None
    embedder = SentenceTransformer(model_id,
                               device="cuda:0")
    print(src_file)
    print(trg_file)
    print(out_file)
    print(df_file)
    src = open(src_file, 'r', encoding='utf-8')
    trg = open(trg_file, 'r', encoding='utf-8')
    tmp_s = []


    tmp_s = load_files(src, trg)

    #67211967
    shuffle(tmp_s)
    df = pd.DataFrame(tmp_s[:max_size], columns=['text', 'trg'])
    del(tmp_s)
    print(df)

    df.to_csv(df_file, index=True, index_label="id", encoding='utf-8')
    del(df)
    gc.collect()
    
    index = train_corpus(df_file, max_size=max_train, embedder=embedder, emb_size=emb_size, nlist=nlist, m=m, nbits=nbits)

    index = index_corpus(index, df_file, out_file=out_file, source_lang=src_lang, embedder=embedder)
    return


if __name__ == '__main__':
     main()



        

