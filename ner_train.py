import torch
import torch
import random
import numpy as np
import pandas as pd
import os
import glob
import torch.backends.cudnn as cudnn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

!git clone https://github.com/kmounlp/NER.git

file_list = []
for x in os.walk('NER/'):
    for y in glob.glob(os.path.join(x[0], '*_NER.txt')):    # ner.*, *_NER.txt
        file_list.append(y)

file_list = sorted(file_list)

!pip install korpora

#NAVER NLP Challenge 2018 Dataset
from Korpora import Korpora
corpus = Korpora.load("naver_changwon_ner")

#한국 해양대학교 데이터셋
from pathlib import Path

file_path = file_list[0]
file_path = Path(file_path)
raw_text = file_path.read_text().strip()
#네이버 NER 챌린지 데이터셋
corpus.train[0:10]

org = pd.read_csv('/kaggle/input/asgasbsa/.csv',)
org_df = org.iloc[:,0:1]

#유명인 이름 데이터셋
per_df = pd.read_csv('/kaggle/input/per-list/per_name.csv')
per_df = per_df.drop(['Unnamed: 0'], axis = 1)
per_df

def naver_read_file(file_list):


    token_docs = []
    tag_docs = []

    for doc in file_list:
        tokens = []
        tags = []
        list1=doc.words
        list2=doc.tags


        # BIO 태깅 방식은 한국 해양대학교 자연어처리 연구실 데이터 셋의 방식으로 통일하기 위해
        # 아래처럼 태깅 방식을 변경
        for text,docs in zip(list1,list2):
            try:
                tag = docs
                if tag == 'ORG_B':
                    tag='B-ORG'
                elif tag == 'PER_B':
                    tag ='B-PER'
                elif tag == 'FLD_B':
                    tag ='B-FLD'
                elif tag == 'AFW_B':
                    tag ='B-AFW'
                elif tag == 'LOC_B':
                    tag ='B-LOC'
                elif tag == 'CVL_B':
                    tag ='B-CVL'
                elif tag == 'DAT_B':
                    tag ='B-DAT'
                elif tag == 'TIM_B':
                    tag ='B-TIM'
                elif tag == 'NUM_B':
                    tag ='B-NUM'
                elif tag == 'EVT_B':
                    tag ='B-EVT'
                elif tag == 'ANM_B':
                    tag ='B-ANM'
                elif tag == 'PLT_B':
                    tag ='B-PLT'
                elif tag == 'MAT_B':
                    tag ='B-MAT'
                elif tag == 'TRM_B':
                    tag ='B-TRM'
                else:
                    tag = 'O'

                if tag in ['B-PER', 'B-DAT', 'B-LOC', 'B-ORG']:
                    if tag == 'B-ORG':
                        token = random.sample(org_df['회사명'].tolist(), k=1)[0] #tag가 B-ORG이면 상장법인목록의 회사명 중 하나로 랜덤으로 대체
                    elif tag == 'B-PER':
                        token = random.sample(per_df['이름'].tolist(), k=1)[0] #tag가 B-PER이면 유명인 이름 리스트 중 하나로 대체
                    else:
                        token = text
                else:
                    token = text
                    tag = 'O'

                for i, syllable in enumerate(token): # 음절 단위로 자르고
                    tokens.append(syllable)
                    modi_tag = tag
                    if i > 0:
                        if tag[0] == 'B':
                            modi_tag = 'I' + tag[1:]     # BIO tag를 부착
                    tags.append(modi_tag)
            except:
                continue
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

naver_text,naver_tags = naver_read_file(corpus.train)
import re


def read_file(file_list):  # 한국 해양대학교
    token_docs = []
    tag_docs = []
    for file_path in file_list:
        # print("read file from ", file_path)
        file_path = Path(file_path)
        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                if line[0:1] == "$" or line[0:1] == ";" or line[0:2] == "##":
                    continue
                try:
                    tag = line.split('\t')[3]  # 2: pos, 3: ner
                    if tag in ['B-PER', 'B-DAT', 'B-LOC', 'B-ORG']:
                        if tag == 'B-ORG':
                            token = random.sample(org_df['회사명'].tolist(), k=1)[0]
                        elif tag == 'B-PER':
                            token = random.sample(per_df['이름'].tolist(), k=1)[0]
                        else:
                            token = line.split('\t')[0]

                        # elif 'I-' in tag:
                    elif tag in ['I-PER', 'I-DAT', 'I-LOC', 'I-ORG']:
                        if tag == 'I-ORG':
                            token = None
                        if tag == 'I-PER':
                            token = None
                        else:
                            token = line.split('\t')[0]
                    else:
                        token = line.split('\t')[0]
                        tag = 'O'

                    # token = line.split('\t')[0]

                    for i, syllable in enumerate(token):  # 음절 단위로
                        tokens.append(syllable)
                        modi_tag = tag
                        if i > 0:
                            if tag[0] == 'B':
                                modi_tag = 'I' + tag[1:]  # BIO tag를 부착
                        tags.append(modi_tag)
                except:
                    continue
            token_docs.append(tokens)
            tag_docs.append(tags)

    return token_docs, tag_docs

texts, tags = read_file(file_list[:])

texts.extend(naver_text)
tags.extend(naver_tags)

unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(list(unique_tags))}
id2tag = {id: tag for tag, id in tag2id.items()}

from sklearn.model_selection import train_test_split
train_texts, test_texts, train_tags, test_tags = train_test_split(texts, tags, test_size=.2, random_state=42)

#여기부터 NER 모델 학습 코드

from transformers import AutoModel, AutoTokenizer, BertTokenizer
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tag2id = {'B-LOC': 0,
 'I-PER': 1,
 'I-ORG': 2,
 'O': 3,
 'B-DAT': 4,
 'I-DAT': 5,
 'B-ORG': 6,
 'I-LOC': 7,
 'B-PER': 8}
unique_tags={'B-DAT',
 'B-LOC',
 'B-ORG',
 'B-PER',
 'I-DAT',
 'I-LOC',
 'I-ORG',
 'I-PER',
 'O'}
id2tag={0: 'B-LOC',
 1: 'I-PER',
 2: 'I-ORG',
 3: 'O',
 4: 'B-DAT',
 5: 'I-DAT',
 6: 'B-ORG',
 7: 'I-LOC',
 8: 'B-PER'}

pad_token_id = tokenizer.pad_token_id # 0
cls_token_id = tokenizer.cls_token_id # 101
sep_token_id = tokenizer.sep_token_id # 102
pad_token_label_id = tag2id['O']    # tag2id['O']
cls_token_label_id = tag2id['O']
sep_token_label_id = tag2id['O']

def ner_tokenizer(sent, max_seq_length):
    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length-2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable  # 중간 음절에는 모두 prefix를 붙입니다.
            # 우리가 구성한 학습 데이터도 이렇게 구성되었기 때문이라고 함.
            # 이순신은 조선 -> [이, ##순, ##신, ##은, 조, ##선]
        pre_syllable = syllable

        input_ids[i] = (tokenizer.convert_tokens_to_ids(syllable))
        attention_mask[i] = 1

    input_ids = [cls_token_id] + input_ids
    input_ids[len(sent)+1] = sep_token_id
    attention_mask = [1] + attention_mask
    attention_mask[len(sent)+1] = 1
    return {"input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids}

tokenized_train_sentences = []
tokenized_test_sentences = []

for text in train_texts:    # 전체 데이터를 tokenizing 합니다.
    tokenized_train_sentences.append(ner_tokenizer(text, 128))
for text in test_texts:
    tokenized_test_sentences.append(ner_tokenizer(text, 128))

def encode_tags(tags, max_seq_length):
    # label 역시 입력 token과 개수를 맞춰줍니다
    tags = tags[:max_seq_length-2]
    labels = [tag2id[tag] for tag in tags]
    labels = [tag2id['O']] + labels

    padding_length = max_seq_length - len(labels)
    labels = labels + ([pad_token_label_id] * padding_length)

    return labels

train_labels = []
test_labels = []

for tag in train_tags:
    train_labels.append(encode_tags(tag, 128))

for tag in test_tags:
    test_labels.append(encode_tags(tag, 128))

import torch

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TokenDataset(tokenized_train_sentences, train_labels)
test_dataset = TokenDataset(tokenized_test_sentences, test_labels)

import accelerate
import transformers

transformers.__version__, accelerate.__version__

# BertForSencenceClassification이 아니다! token이 목적이다.
from transformers import BertForTokenClassification, Trainer, TrainingArguments, AutoModelForTokenClassification,EarlyStoppingCallback
import sys
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1000, # 1000번쨰 steps마다 log를 보여줌
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=5,
    save_strategy='steps', # steps로 해야 earlystop이 가능
    evaluation_strategy='steps',
    save_steps=1000, # 1000번쨰 step마다 저장
    eval_steps=1000, # 1000번째 step마다 평가
    seed=15,
    load_best_model_at_end=True # 가장 좋은 성능의 모델로...
)

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_tags))
model.to(device)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,# evaluation dataset
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] #loss가 2번 감소하지 않으면 스탑
)

import gc
gc.collect()

trainer.train()
trainer.evaluate()

predictions = trainer.predict(test_dataset)
print(predictions.predictions.shape, predictions.label_ids.shape)

preds = np.argmax(predictions.predictions, axis=-1)
index_to_ner = {i:j for j, i in tag2id.items()}
f_label = [i for i, j in tag2id.items()]
val_tags_l = [index_to_ner[x] for x in np.ravel(predictions.label_ids).astype(int).tolist()]
y_predicted_l = [index_to_ner[x] for x in np.ravel(preds).astype(int).tolist()]

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print(classification_report(val_tags_l, y_predicted_l, labels=f_label))

trainer.save_model('ner_model')

# 저장한 모델 불러오기
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from transformers import AutoModel, AutoTokenizer, BertTokenizer
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
unique_tags={'B-DAT',
 'B-LOC',
 'B-ORG',
 'B-PER',
 'I-DAT',
 'I-LOC',
 'I-ORG',
 'I-PER',
 'O'}
#tag2id와 id2tag는 학습하며 지정된 그대로 사용

# tag2id = {'B-LOC': 0,
#  'I-PER': 1,
#  'I-ORG': 2,
#  'O': 3,
#  'B-DAT': 4,
#  'I-DAT': 5,
#  'B-ORG': 6,
#  'I-LOC': 7,
#  'B-PER': 8}
# id2tag={0: 'B-LOC',
#  1: 'I-PER',
#  2: 'I-ORG',
#  3: 'O',
#  4: 'B-DAT',
#  5: 'I-DAT',
#  6: 'B-ORG',
#  7: 'I-LOC',
#  8: 'B-PER'}

pad_token_id = tokenizer.pad_token_id # 0
cls_token_id = tokenizer.cls_token_id # 101
sep_token_id = tokenizer.sep_token_id # 102
pad_token_label_id = tag2id['O']    # tag2id['O']
cls_token_label_id = tag2id['O']
sep_token_label_id = tag2id['O']

model = AutoModelForTokenClassification.from_pretrained('/kaggle/input/ner-per-comp', num_labels=len(unique_tags))
model.to(device)

# 기존 토크나이저는 wordPiece tokenizer로 tokenizing 결과를 반환합니다.
# 데이터 단위를 음절 단위로 변경했기 때문에, tokenizer도 음절 tokenizer로 변경

# berttokenizer를 사용하는데 한국어 vocab이 8000개 정도 밖에 없고 그 안의 한국어들의 거의 음절로 존재
# -> 음절 단위 tokenizer를 적용하면 vocab id를 어느 정도 획득할 수 있어 UNK가 별로 없을듯 하다
def ner_tokenizer(sent, max_seq_length):
    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length-2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable  # 중간 음절에는 모두 prefix를 붙입니다.
            # 우리가 구성한 학습 데이터도 이렇게 구성되었기 때문이라고 함.
            # 이순신은 조선 -> [이, ##순, ##신, ##은, 조, ##선]
        pre_syllable = syllable

        input_ids[i] = (tokenizer.convert_tokens_to_ids(syllable))
        attention_mask[i] = 1

    input_ids = [cls_token_id] + input_ids
    input_ids[len(sent)+1] = sep_token_id
    attention_mask = [1] + attention_mask
    attention_mask[len(sent)+1] = 1
    return {"input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids}

#Inference
def ner_inference(text) :

    model.eval()
    text = text.replace(' ', '_')

    predictions , true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text)+2)
    input_ids = torch.tensor(tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.append(label_ids)

    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    print('{}\t{}'.format("TOKEN", "TAG"))
    print("===========")
    # for token, tag in zip(tokenizer.decode(tokenized_sent['input_ids']), pred_tags):
    #   print("{:^5}\t{:^5}".format(token, tag))
    for i, tag in enumerate(pred_tags):
        print("{:^5}\t{:^5}".format(tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]), tag))


text = '박정규는 12월 25일에 나아지는 성능을 보며 SK기업의 후원을 받는 임주원씨를 기다리고 있다.'
ner_inference(text)

text = '종목별로는  미래에셋증권(+2.83%)  메리츠증권(+3.99%)  한국금융지주(+3.07%)  삼성증권(+2.67%)  NH투자증권(+2.08)  키움증권(+4.06%) 등이 강세를 보였다. KRX 증권 지수는 증시에 상장된 증권업종의 주가 흐름을 반영하는 지수로 미래에셋증권, 한국금융지주, NH투자증권 등 14개 종목이 지수에 포함돼 있다. 증권정보업체 에프앤가이드에 따르면 실적추정치가 있는 증권사 다섯 군데(삼성증권, 미래에셋증권, 키움증권, 한국금융지주, NH투자증권)의 4분기 영업이익 전망치 합은 8558억 원으로 전년 동기보다 27.60% 줄어들 전망이다.'

ner_inference(text)


#아래는 NER 대상 데이터 EDA

import numpy as np
import matplotlib.pyplot as plt

texts_len = [len(x) for x in texts]

plt.figure(figsize=(16,10))
plt.hist(texts_len, bins=50, range=[0,800], facecolor='b', density=True, label='Text Length')
plt.title('Text Length Histogram')
plt.legend()
plt.xlabel('Number of Words')
plt.ylabel('Probability')

#각 NER 태그별 데이터 개수

for tag in list(tag2id.keys()) :
    globals()[tag] = 0
for tag in tags :
    for ner in tag :
        globals()[ner] += 1
for tag in list(tag2id.keys()) :
    print('{:>6} : {:>7,}'. format(tag, globals()[tag]))



