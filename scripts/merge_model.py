from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "google/gemma-3-4b-it"
lora = "models/gemma3-4b-qlora_2epoch"
merge_out = "models/gemma3-4b_merge-qlora_2epoch" 

print(merge_out)

# 1. Load the base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base, 
    device_map="auto", 
    torch_dtype="auto" # Avoid 4-bit or 8-bit quantization here
)
tokenizer = AutoTokenizer.from_pretrained(base)

# 2. Load the PEFT/LoRA adapter
model = PeftModel.from_pretrained(base_model, lora)

# 3. Merge weights and unload the adapter
merged_model = model.merge_and_unload()

# 4. Save the merged model and tokenizer
merged_model.save_pretrained(merge_out)
tokenizer.save_pretrained(merge_out)