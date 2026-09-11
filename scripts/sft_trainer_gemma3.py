# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
import transformers
from trl import SFTTrainer, SFTConfig
#from trl import setup_chat_format, clone_chat_template
import os
import argparse
import logging

import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="SFT trainer Gemma3")

    parser.add_argument(
        "--data",
        type=str,
        default="combo_data/en-pl_es-eu.eng-tech_health.term-aya.5-shot.train-hf-sample.json",
        help="data combination of language pairs and domains",
    )

    parser.add_argument(
        "--output_dir ",
        type=str,
        default="models/gemma3-4b-qlora_2epoch",
        help="",
    )


    parser.add_argument(
            "--model_id",
            type=str,
            default="google/gemma-3-4b-it",
            help="",
            )  


    parser.add_argument(
            "--train_bs",
            type=int,
            default=2,
            help="",
            )

    parser.add_argument(
            "--grad_acc",
            type=int,
            default=4,
            help="",
            )

    parser.add_argument(
            "--lr",
            type=float,
            default=2e-4,
            help="",
            )

    parser.add_argument(
            "--w_steps",
            type=float,
            default=0.03,
            help="",
            )

    parser.add_argument(
            "--e_epoch",
            type=int,
            default=2,
            help="",
            )

    parser.add_argument(
            "--lr_scheduler_type",
            type=str,
            default="cosine",
            help="",
            )

    parser.add_argument(
            "--lora_r",
            type=int,
            default=32,
            help="",
            )

    parser.add_argument(
            "--lora_alpha",
            type=int,
            default=64,
            help="",
            )

    parser.add_argument(
            "--lora_dropout",
            type=float,
            default=0.1,
            help="",
            )

    
    
    args = parser.parse_args()

    return args

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



def main():

    args = parse_args()

    model_id = args.model_id # "google/gemma-3-4b-it" #"Qwen/Qwen3-8B"   
    #CUDA_VISIBLE_DEVICES=1 accelerate launch sft_trainer_gemma3.py
    max_length = args.max_length #1024
    data_files = {}
    data_files["train"] = args.data #"combo_data/en-pl_es-eu.eng-tech_health.term-aya.5-shot.train-hf-sample.json"    
    output_dir = args.output_dir #"models/gemma3-4b-qlora_2epoch" #'models/tyni-aya-3b-qlora4bit_peft_finetune'

    train_bs = args.train_bs #2
    grad_acc = args.gradd_acc #4
    lr = args.lr #2e-4 #2e-5
    w_steps = args.w_steps #0.03
    n_epoch = args.n_epoch #2
    lr_scheduler_type = args.lr_scheduler_type #"cosine"
    lora_r = args.lora_r #32
    lora_alpha = args.lora_alpha #64
    lora_dropout = args.lora_dropout #0.1
    #system_message = None

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,   #qlora
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
                 #BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                 #dtype="auto", #8bit
                                                 device_map={"": 0}) #{"": 0}


    model = prepare_model_for_kbit_training(model)

   
    #LoRA
    #model.gradient_checkpointing_enable()
    #https://medium.com/@_mishy/how-to-use-gradient-checkpointing-with-hugging-face-models-9821bd8b51b1
    config = LoraConfig( #BEST MODEL!!
          r=lora_r,
          lora_alpha=lora_alpha,
          lora_dropout=lora_dropout, #0.05
          bias="none",
          task_type="CAUSAL_LM",
          target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
      )

    print(model)

    def create_conversation(sample):
        #_, trg_id = sample["translation"]['lang_pair']
        prompt = sample["translation"]["prompt"]
        trg = sample["translation"]['trg']       
        msg = {
            "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},  #[{"type": "text", "text": prompt}]
                    {"role": "assistant", "content": [{"type": "text", "text": trg}]}
                ]
            }
        #print(msg)
        return msg
 


    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "right"



    data = load_dataset("json", data_files=data_files, encoding='utf-8')
    #data = data.shuffle(seed=42)
   

    data = data.map(create_conversation, batched=False)
    data = data.remove_columns("translation")



    args = SFTConfig(per_device_train_batch_size=train_bs,
                    gradient_accumulation_steps=grad_acc,
                    warmup_ratio=w_steps,
                    lr_scheduler_type=lr_scheduler_type,
                    num_train_epochs=n_epoch,
                    learning_rate=lr,
                    save_total_limit=1,
                    save_strategy="epoch",
                    output_dir=output_dir,
                    optim="paged_adamw_8bit",
                    max_length=max_length,
                    push_to_hub=False,
                    report_to='none'
                    )

    trainer = SFTTrainer(model=model,
                         args=args,
                         train_dataset=data['train'],
                         peft_config=config
                         )

    
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return




if __name__ == '__main__':
     main()
