# huggingface-cli login --token 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 # your code
import os
import torch
import transformers
from datasets import load_from_disk
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TextStreamer,
    pipeline
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
    PeftModel
)
from trl import SFTTrainer
from peft import PeftModel, PeftConfig
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL = "Qwen2.5-1.5B-Instruct_real"
device_map={"": 0}

peft_config = PeftConfig.from_pretrained(FINETUNED_MODEL)


# NF4 양자화를 위한 설정
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, # 모델을 4비트 정밀도로 로드
    bnb_4bit_quant_type="nf4", # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
    bnb_4bit_use_double_quant=True, # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
    bnb_4bit_compute_dtype=torch.bfloat16 # 연산 속도를 높이기 위해 사용 (default: torch.float32)
)
# 베이스 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=nf4_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    cache_dir='./model'
)
tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path
)

# QLoRA 모델 로드
peft_model = PeftModel.from_pretrained(model, FINETUNED_MODEL, torch_dtype=torch.bfloat16)
# QLoRA 가중치를 베이스 모델에 병합
merged_model = peft_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
prompt = "농구를 하다가 손이 골절된 것 같아요. 많이 아파서 제대로 움직일 수가 없습니다."

# 텍스트 생성을 위한 파이프라인 설정
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, max_new_tokens=256)
outputs = pipe(
    prompt,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.2,
    add_special_tokens=True
)
print(outputs[0]["generated_text"][len(prompt):])

prompt = "밤에 다리가 따끔거리고 불편한 감각이 느껴져서 잠을 설치고 있습니다. 그것 때문에 잠을 잘 수가 없어요."
outputs = pipe(
    prompt,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.2,
    add_special_tokens=True
)
print(outputs[0]["generated_text"][len(prompt):])
