# QLora 기법을 활용한 KULLUM3 모델 파인튜닝

Github : [https://github.com/ezreal12/qlora_fine_tuning](https://github.com/ezreal12/qlora_fine_tuning)

## 목차

# 초록

---

본 연구에서는 한국어 대규모 언어 모델인 KULLM3를 의료 도메인에 특화된 모델로 파인튜닝하는 과정을 제시한다. 이를 위해 QLoRA (Quantized Low-Rank Adaptation) 기법을 적용하여 메모리 효율성을 높이고, 4비트 양자화를 통해 계산 효율성을 개선하였다. MedGPT-5k-ko 데이터셋을 사용하여 모델을 학습시켰으며, 파인튜닝 결과 의료 관련 질문에 대해 적절한 응답을 생성할 수 있음을 확인하였다. 본 연구의 방법론은 제한된 컴퓨팅 자원으로도 대규모 언어 모델을 효과적으로 특정 도메인에 적응시킬 수 있음을 보여준다.

# 1. 서론

---

최근 대규모 언어 모델(Large Language Models, LLMs)의 발전으로 다양한 자연어 처리 작업에서 획기적인 성능 향상이 이루어지고 있다. 그러나 이러한 모델들의 크기와 복잡성으로 인해 특정 도메인이나 작업에 맞게 모델을 조정하는 것이 큰 도전 과제가 되고 있다. 특히, 전체 모델 파라미터를 미세 조정(fine-tuning)하는 방식은 막대한 컴퓨팅 자원과 시간을 필요로 한다.
이러한 문제를 해결하기 위해 파라미터 효율적 미세 조정(Parameter Efficient Fine-Tuning, PEFT) 기법들이 제안되었다. 그 중에서도 LoRA (Low-Rank Adaptation)와 이를 개선한 QLoRA (Quantized LoRA) 기법은 모델의 성능을 유지하면서도 훨씬 적은 자원으로 효과적인 파인튜닝을 가능하게 한다.
본 연구에서는 한국어 대규모 언어 모델인 KULLM3를 기반으로 하여 의료 도메인에 특화된 모델로 파인튜닝하는 과정을 제시한다. 이를 위해 QLoRA 기법을 적용하여 메모리 효율성을 높이고, 4비트 양자화를 통해 계산 효율성을 개선하였다.
연구의 개발 환경으로는 Google Colaboratory의 A100 GPU를 사용하였으며, 시스템 RAM 83.5GB, GPU RAM 40.0GB의 하드웨어 스펙을 활용하였다. 이러한 환경에서 QLoRA 기법을 통해 대규모 언어 모델을 효율적으로 파인튜닝할 수 있음을 보여주고자 한다. 특히, 제한된 컴퓨팅 자원으로도 특정 도메인에 특화된 언어 모델을 구축할 수 있는 방법론을 제시함으로써, 다양한 분야에서의 언어 모델 활용 가능성을 확장하고자 한다.
본 논문의 구성은 다음과 같다. 2장에서는 PEFT 기법의 개요와 주요 이점에 대해 설명한다. 3장에서는 LoRA 기법의 원리와 구현 방식에 대해 상세히 다룬다. 4장에서는 QLoRA 기법의 특징과 장점을 설명한다. 5장에서는 본 연구에서 사용한 KULLM3 모델에 대해 소개한다. 6장에서는 파인튜닝에 사용된 MedGPT-5k-ko 데이터셋에 대해 설명한다. 7장에서는 KULLM3 모델의 파인튜닝 과정을 단계별로 상세히 기술한다. 마지막으로 8장에서는 연구의 결론과 향후 연구 방향을 제시한다.

# 2. PEFT (Parameter Efficient Fine-Tuning) 기법

---

## 2.1 PEFT의 개요

PEFT는 대규모 언어 모델을 효율적으로 미세 조정하기 위한 기법들의 집합이다. 이 기법은 전체 모델을 미세 조정하는 것과 비슷한 성능을 유지하면서도 계산 비용과 시간을 크게 절감할 수 있다. 특히 수십억 개의 매개변수를 가진 거대 모델의 경우, 전체 미세 조정에 막대한 비용이 소요되므로 PEFT의 필요성이 더욱 부각된다.

## 2.2 PEFT의 주요 이점

- 계산 효율성: 전체 모델 대신 일부 매개변수만 조정하여 학습 시간과 비용을 절감한다.
- 성능 유지: 전체 미세 조정에 준하는 성능을 달성할 수 있다.
- 유연성: 다양한 작업에 대해 동일한 기본 모델을 사용하면서 작업별로 특화된 조정이 가능하다.

## 2.3 주요 PEFT 기법

PEFT에는 여러 기법이 포함되며, 그 중 대표적인 것으로는 Adapters, LoRA (Low-Rank Adaptation), 그리고 Prompt Tuning 등이 있다. 이 기법들은 각각 고유한 방식으로 효율적인 미세 조정을 달성한다.

# 3. LoRA (Low-Rank Adaptation) 기법

---

## 3.1 LoRA의 원리

LoRA는 신경망의 가중치 행렬이 낮은 본질적 차원(intrinsic dimension)을 가진다는 관찰에 기반한다. 이는 미세 조정 과정에서 가중치 업데이트 행렬이 낮은 랭크를 가질 수 있음을 의미한다. LoRA는 이 특성을 활용하여 가중치 업데이트를 효율적으로 수행한다.

## 3.2 LoRA의 구현 방식

- 가중치 업데이트 행렬(ΔW)을 두 개의 작은 행렬(A와 B)로 분해한다.
- 분해된 행렬의 랭크(r)를 조절하여 학습 가능한 매개변수의 수를 제어한다.
- 원본 모델의 가중치는 동결시키고, 분해된 행렬만 학습한다.

## 3.3 LoRA의 효율성

- 매개변수 효율성: 전체 매개변수의 약 2%만으로도 전체 미세 조정에 준하는 성능을 달성할 수 있다.
- 저장 효율성: 학습된 LoRA 가중치는 매우 작은 크기로 저장 가능하다.
- 유연성: 특정 레이어에만 LoRA를 적용하여 더욱 효율적인 학습이 가능하다.

## 3.4 LoRA의 응용

LoRA는 텍스트 생성 모델뿐만 아니라 이미지 생성 모델에서도 효과적으로 사용된다. 특히 Stable Diffusion과 같은 모델에서 LoRA를 통해 특정 스타일이나 주제에 대한 학습이 가능하며, 여러 LoRA 가중치를 조합하여 다양한 스타일을 생성할 수 있다.

# 4. QLoRA (Quantized LoRA)

---

## 4.1 QLoRA의 개념

QLoRA는 LoRA의 확장된 버전으로, 모델 양자화 기법을 LoRA와 결합하여 메모리 효율성을 더욱 향상시킨 방법이다. 이 기법은 기존 LoRA의 이점을 유지하면서도 메모리 사용량을 크게 줄일 수 있다.

## 4.2 QLoRA의 주요 특징

- 4비트 양자화: 모델의 가중치를 4비트로 양자화하여 메모리 사용량을 대폭 감소시킨다.
- 페이저 양자화: 양자화 오차를 줄이기 위한 특별한 양자화 기법을 사용한다.
- 메모리 효율성: QLoRA를 통해 기존 LoRA보다 더 적은 메모리로 대규모 언어 모델을 미세 조정할 수 있다.

## 4.3 QLoRA의 장점

- 단일 GPU 학습: 대규모 언어 모델을 단일 GPU에서도 효과적으로 학습할 수 있게 한다.
- 성능 유지: 양자화에도 불구하고 전체 정밀도 미세 조정과 비슷한 수준의 성능을 유지한다.
- 확장성: 더 큰 모델이나 더 많은 데이터에 대한 미세 조정을 가능하게 한다.

# 5. LLM 모델 선택 - KULLM3

---

## 5.1 LLM 모델 선택

모델을 파인튜닝 할때 LLama 3, Mistral 3B등 여러 모델을 고려할 수 있지만 이번 예시에서는 한국어 데이터셋으로 파인튜닝 된 한국어 특화 모델인 KULLM3를 선택하였다.

## 5.2 KULLM3 모델 개요

KULLM3은 고려대학교 NLP & AI 연구실과 HIAI 연구소가 개발한 한국어 Large Language Model (LLM)이다. 이 모델은 고급 지시 따르기 능력과 유창한 대화 능력을 갖춘 것으로 알려져 있으며, 특히 gpt-3.5-turbo의 성능을 근접하게 따라가는 것으로 나타났다. KULLM3은 현재 공개된 한국어 언어 모델 중 최고 수준의 성능을 보이는 모델 중 하나로 평가받고 있다.

## 5.3 모델 특성

- 기반 모델: upstage/SOLAR-10.7B-v1.0을 기반으로 미세 조정되었다.
- 언어 지원: 한국어와 영어를 지원한다.
- 라이선스: Apache 2.0 라이선스로 제공된다.

## 5.4 학습 데이터 및 절차

KULLM3의 학습에는 다음과 같은 데이터와 절차가 사용되었다:

- 학습 데이터:
    - vicgalle/alpaca-gpt4 데이터셋
    - 혼합 한국어 지시 데이터 (GPT 생성, 수작업 제작 등)
    - 총 약 66,000개 이상의 예제 사용
- 학습 절차:
    - 고정된 시스템 프롬프트를 사용하여 학습되었다.
    - 시스템 프롬프트는 모델의 정체성, 윤리적 가이드라인, 대화 스타일 등을 정의한다.

# 6. MedGPT-5k-ko 데이터셋

---

## 6.1 데이터셋 선택

데이터셋은 huggingface를 통해 제공받은 MedGPT-5k-ko 데이터셋을 이용하였다. MedGPT-5k-ko 데이터셋은 한국어로 작성된 환자의 질문 - 의사의 답변 형식의 데이터셋으로 의료 분야에 한정된 데이터셋을 이용해 의료 분야 도메인 특화 파인튜닝이 가능하다. 또한, 상대적으로 적은 양의 데이터로 GPU Out Of Memory 오류를 방지하고 빠른 파인튜닝 결과 확인이 가능하다.

## 6.2 데이터셋 개요

MedGPT-5k-ko는 한국어 의료 대화 데이터셋으로, 의료 분야의 자연어 처리 및 대화 시스템 개발을 위해 구축되었다. 이 데이터셋은 의사와 환자 간의 대화를 시뮬레이션한 5,304개의 대화 쌍으로 구성되어 있으며, 다양한 의료 상황과 증상에 대한 질문과 응답을 포함하고 있다. 한국어로 작성된 의료 전문분야의 데이터를 포함하고 있어, 도메인 특화된 모델을 만들기 위한 파인튜닝에 적합한 데이터셋이다. 또한, 상대적으로 적은 크기의 데이터셋이기 때문에 파인튜닝 결과를 빠르게 확인할 수 있어 실험 및 연구에 효율적으로 활용할 수 있다.

## 6.3 데이터셋 특성

- 크기: 5,304개의 대화 쌍
- 언어: 한국어
- 형식: JSON
- 라이선스: GPL-3.0
- 총 파일 크기: 2.38 MB (다운로드 시), 861 kB (Parquet 변환 후)

# 7. KULLM3 모델의 파인튜닝 과정

---

본 연구에서는 KULLM3 모델을 기반으로 하여 의료 도메인에 특화된 모델로 파인튜닝을 진행하였다. 이 과정에서 QLoRA (Quantized Low-Rank Adaptation) 기법을 적용하여 메모리 효율성을 높이고, 4비트 양자화를 통해 계산 효율성을 개선하였다. 파인튜닝 과정의 주요 단계는 다음과 같다.

## 7.1 기본 모델 및 데이터셋 준비

먼저, KULLM3  모델과 해당 모델의 토크나이저를 로드하였다. 데이터셋으로는 mncai/MedGPT-5k-ko를 사용하였다.

```python
BASE_MODEL = "nlpai-lab/KULLM3"
device_map={"": 0}
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map=device_map, cache_dir= './model')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir= './model')

dataset = load_dataset('mncai/MedGPT-5k-ko', split="train", cache_dir='./load_dataset')

```

## 7.2 모델 양자화

BitsAndBytesConfig를 사용하여 4비트 NormalFloat (NF4) 양자화를 적용하였다. 이 과정에서 이중 양자화 기법을 사용하여 메모리 효율성을 향상시켰고, bfloat16 dtype을 사용하여 연산 속도를 개선하였다. bfloat16 dtype은 모델의 학습 속도를 향상시키지만, 고가의 NVDIA GPU만 지원하는 기능이기 때문에 GPU 호환 여부를 체크해야한다. 대표적으로 A100이 있다. bfloat16 dtype을 지원하지 않는다면 torch.float16을 이용하면 된다.

```python
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=nf4_config,
    device_map=device_map,
    cache_dir='./model'
)

```

## 7.3 프롬프트 템플릿 설정

입력이 있는 경우와 없는 경우에 대한 두 가지 프롬프트 템플릿을 정의하였다. 이 템플릿은 지시사항, 입력, 응답의 구조로 구성되어 있다.

```python
prompt_input_template = """아래는 작업을 설명하는 지시사항과 추가 정보를 제공하는 입력이 짝으로 구성됩니다. 이에 대한 적절한 응답을 작성해주세요.

### 지시사항:
{instruction}

### 입력:
{input}

### 응답:"""

prompt_no_input_template = """아래는 작업을 설명하는 지시사항입니다. 이에 대한 적절한 응답을 작성해주세요.

### 지시사항:
{instruction}

### 응답:"""

```

## 7.4 데이터 전처리

정의된 프롬프트 템플릿을 사용하여 데이터셋의 각 예시에 프롬프트를 적용하고, 토크나이저를 사용하여 텍스트를 토큰화하였다. 최대 길이는 512로 제한하였다.

```python
def generate_prompt(data_point):
    # 프롬프트 생성 로직
    ...

dataset_cvted = dataset.shuffle().map(generate_prompt, remove_columns=remove_column_keys)

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, max_length=512)
    return outputs

dataset_tokenized = dataset_cvted.map(tokenize_function, batched=True, remove_columns=dataset_cvted.features.keys())

```

## 7.5 LoRA 설정

LoRA 구성을 위해 LoraConfig를 사용하였다. 이 설정에서는 LoRA 가중치 행렬의 rank (r)를 4로, 스케일링 팩터 (lora_alpha)를 8로 설정하였다.

```python
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    bias='none',
    task_type=TaskType.CAUSAL_LM
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

```

## 7.6 학습 설정

TrainingArguments를 사용하여 학습 설정을 구성하였다. 배치 크기는 2, 그래디언트 누적 단계는 4, 학습 에폭은 3으로 설정하였다. 옵티마이저로는 8비트 AdamW를 사용하였다. Colab에서의 한정된 하드웨어 이용으로 인해 학습 step 수를 1000으로 제한하여 빠른 결과 확인이 가능하게 하였다.

```python
train_args = transformers.TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=1,
    max_steps=1000,
    learning_rate=2e-4,
    bf16=False,
    output_dir="outputs",
    optim="paged_adamw_8bit",
    logging_steps=50,
    save_total_limit=3
)

```

## 7.7 학습 실행

SFTTrainer를 사용하여 모델 학습을 실행하였다. 최대 시퀀스 길이는 512로 설정하였고, 커스텀 데이터 콜레이터 함수를 사용하여 배치 처리를 수행하였다.

```python
def collate_fn(examples):
    examples_batch = tokenizer.pad(examples, padding='longest', return_tensors='pt')
    examples_batch['labels'] = examples_batch['input_ids']
    return examples_batch

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_tokenized,
    max_seq_length=512,
    args=train_args,
    dataset_text_field="text",
    data_collator=collate_fn
)

model.config.use_cache = False
trainer.train()

```

![tarin](https://github.com/user-attachments/assets/f3d888a5-4ee5-4ade-9c03-c2627b133477)

TrainingArguments에서 설정한 대로 1000회 step 학습하여 0.2766까지 Training Loss를 줄일 수 있었으며 Colab A100 하드웨어에서 1000 step 학습까지 총 1시간 7분정도가 소요되었다.

## 7.8 모델 저장

파인튜닝이 완료된 모델을 임의의 디렉토리에 저장하였다. 이렇게 저장된 모델은 이번에 학습된 QLora 레이어만 파일로 저장하게 되며, 추후 모델 이용시에 베이스 모델과 합쳐 사용하는 과정이 필요하다.

```python
FINETUNED_MODEL = "QLora_KULLM3"
trainer.model.save_pretrained(FINETUNED_MODEL)

```

## 7.9 모델 로드 테스트

파인튜닝된 QLora 모델의 설정을 정의하고 모델을 로드한다. 모델을 Load 할때의 설정은 특이사항이 없다면 파인튜닝 당시와 동일하게 하면 되며 GPU의 bfloat16 dtype 미지원등 환경에 변화가 생겼다면 임의로 수정하면 된다.

```python
from peft import PeftModel, PeftConfig
peft_config = PeftConfig.from_pretrained(FINETUNED_MODEL)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=nf4_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path
)

```

다음으로, QLoRA 모델을 로드하고 이를 기본 모델과 병합하여 파인튜닝한 모델을 테스트 할 준비를 마친다.

```python
peft_model = PeftModel.from_pretrained(model, FINETUNED_MODEL, torch_dtype=torch.bfloat16)
merged_model = peft_model.merge_and_unload()

```

```python
BASE_MODEL = "nlpai-lab/KULLM3"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
prompt = "농구를 하다가 손이 골절된 것 같아요. 많이 아파서 제대로 움직일 수가 없습니다."

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

```

![test](https://github.com/user-attachments/assets/c218e332-9979-4036-ade5-fbd9aff5c5bd)


모델이 의료 관련 분야의 질문에도 답변이 가능한것을 확인하였다.

# 결론

---

본 연구에서는 한국어 대규모 언어 모델인 KULLM3를 의료 도메인에 특화된 모델로 파인튜닝하는 과정을 제시하였다. QLoRA 기법과 4비트 양자화를 활용하여 메모리 및 계산 효율성을 크게 향상시켰으며, 이를 통해 단일 GPU 환경에서도 대규모 언어 모델의 파인튜닝이 가능함을 보였다.

본 연구의 주요 결과는 다음과 같다:

1. KULLM3 모델을 의료 도메인에 특화된 모델로 성공적으로 파인튜닝하였다. 이는 QLoRA 기법과 4비트 양자화를 통해 가능했으며, 이로 인해 메모리 사용량을 크게 줄이면서도 모델의 성능을 유지할 수 있었다.
2. MedGPT-5k-ko 데이터셋을 사용하여 의료 분야에 특화된 파인튜닝을 수행하였다. 이 데이터셋은 한국어 의료 대화로 구성되어 있어, 모델이 의료 관련 질문에 적절히 응답할 수 있도록 하는 데 크게 기여하였다.
3. 1000 스텝의 학습 후, 훈련 손실(Training Loss)이 0.2766까지 감소하였으며, 이는 모델이 의료 도메인의 특성을 잘 학습했음을 시사한다.
4. 파인튜닝된 모델을 사용하여 의료 관련 질문에 대한 응답을 생성하는 테스트를 수행하였고, 모델이 적절하고 정확한 의료 정보를 제공할 수 있음을 확인하였다.

본 연구의 접근 방식은 다음과 같은 의의를 가진다:

1. 리소스 효율성: QLoRA와 4비트 양자화를 통해 대규모 언어 모델을 효율적으로 파인튜닝할 수 있음을 보여주었다. 이는 고성능 컴퓨팅 자원에 대한 접근성이 제한된 연구자들에게도 대규모 언어 모델을 활용할 수 있는 가능성을 제시한다.
2. 도메인 특화: 의료 분야에 특화된 데이터셋을 사용하여 모델을 파인튜닝함으로써, 특정 도메인에 대한 언어 모델의 성능을 향상시킬 수 있음을 입증하였다. 이는 다양한 전문 분야에서 언어 모델의 활용 가능성을 넓힌다.
3. 한국어 모델 개선: KULLM3라는 한국어 특화 모델을 기반으로 하여, 한국어 의료 대화 시스템의 성능을 향상시켰다. 이는 한국어 자연어 처리 분야의 발전에 기여한다.
4. 실용적 응용: 파인튜닝된 모델이 실제 의료 관련 질문에 적절히 응답할 수 있음을 보여줌으로써, 이러한 기술이 의료 상담 보조 시스템 등 실제 응용 분야에서 활용될 수 있는 가능성을 제시하였다.

향후 연구 방향으로는 다음과 같은 사항들을 고려할 수 있다:

1. 더 큰 규모의 의료 데이터셋을 사용하여 모델의 성능을 더욱 향상시키는 연구
2. 자유롭게 이용 가능한 하드웨어에서 더욱 시간을 들인 학습, 모델이 과적합 되지 않도록 드롭아웃 하는 연구
3. 데이터셋을 보강하여 다양한 의료 세부 분야(예: 내과, 외과, 정신과 등)에 특화된 모델 개발
4. 실제 의료 현장에서의 활용을 위한 사용자 인터페이스 및 시스템 통합 연구
5. 다국어 의료 대화 모델로의 확장을 통한 글로벌 의료 정보 접근성 향상

결론적으로, 본 연구는 QLoRA 기법을 활용한 KULLM3 모델의 파인튜닝을 통해 효율적이고 효과적인 의료 도메인 특화 언어 모델 개발 방법을 제시하였다. 이러한 접근 방식은 한정된 컴퓨팅 자원으로도 대규모 언어 모델을 특정 도메인에 적용할 수 있는 가능성을 보여주며, 향후 다양한 분야에서의 언어 모델 활용 연구에 기여할 것으로 기대된다.

# 참고자료

---

[**Guide to fine-tuning LLMs using PEFT and LoRa techniques**]

[https://www.mercity.ai/blog-post/fine-tuning-llms-using-peft-and-lora](https://www.mercity.ai/blog-post/fine-tuning-llms-using-peft-and-lora)

[**In-depth guide to fine-tuning LLMs with LoRA and QLoRA**]

[https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora](https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora)

[Llama2 4bit QLoRA with kullm]

[https://colab.research.google.com/drive/19AFEOrCI6-bc7h9RTso_NndRwXJRaJ25](https://colab.research.google.com/drive/19AFEOrCI6-bc7h9RTso_NndRwXJRaJ25)

[nlpai-lab/KULLM3 huggingface]

[https://huggingface.co/nlpai-lab/KULLM3](https://huggingface.co/nlpai-lab/KULLM3)

[mncai/MedGPT-5k-kohuggingface]

[https://huggingface.co/datasets/mncai/MedGPT-5k-ko](https://huggingface.co/datasets/mncai/MedGPT-5k-ko)
