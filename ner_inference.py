import json
import os
import csv
import random
import pandas as pd
import torch
from transformers import pipeline, AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

#ner
from transformers import AutoModel, AutoTokenizer, BertTokenizer
import torch.backends.cudnn as cudnn

# torch.cuda.get_device_name(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device

import os
os.chdir('/content/drive/MyDrive/Colab Notebooks/aiku/project/플젝학습')

# file_path = '/content/drive/MyDrive/nlp_data/datas/all_aug_data.csv'
file_path = './all_aug_data.csv'

data = pd.read_csv(file_path)

data.head()

num = int(len(data) * 0.8)
data = data.sample(frac=1, random_state = 42).reset_index(drop=True)
train_data = data[:num]
test_data = data[num:]

print(len(train_data),len(test_data))

!pip install koalanlp

ner_train_data = train_data.fillna('') #content 결측값 예외처리

import kss

splitter = kss.split_sentences

sentence_list=[]
index_list=[]
for i in tqdm(range(ner_train_data.shape[0])):

    s_text=splitter(ner_train_data.content[i])
    sentence_list.append(s_text)
    count=len(s_text)
    index_list.append(count)


content_sentence_list=[]
# content_sentence_list = ner_train_data.content
content_sentence_list = sentence_list

title_sentence_list = []
title_sentence_list = ner_train_data.title

print(len(content_sentence_list))
print(len(title_sentence_list))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from transformers import AutoModel, AutoTokenizer, BertTokenizer
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

'''
새 태그
id2tag
{0: 'B-LOC',
 1: 'I-PER',
 2: 'I-ORG',
 3: 'O',
 4: 'B-DAT',
 5: 'I-DAT',
 6: 'B-ORG',
 7: 'I-LOC',
 8: 'B-PER'}

tag2id
{'B-LOC': 0,
 'I-PER': 1,
 'I-ORG': 2,
 'O': 3,
 'B-DAT': 4,
 'I-DAT': 5,
 'B-ORG': 6,
 'I-LOC': 7,
 'B-PER': 8}
unique_tags
{'B-DAT',
 'B-LOC',
 'B-ORG',
 'B-PER',
 'I-DAT',
 'I-LOC',
 'I-ORG',
 'I-PER',
 'O'}
'''

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

model = AutoModelForTokenClassification.from_pretrained('./ner_per_comp_model', num_labels=len(unique_tags))
model.to(device)

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

# 우리가 전에 사용했던건 word piece tokenizer
# 지금 사용한건 음절 단위 tokenizer
# 음절 tokenizer를 거친 후에 model에 들어가야 한다.

def ner_inference(text) :
    words = []
    tags= []
    model.eval()
    text = text.replace(' ', '_')
    #text = text[:510]

    predictions, true_labels = [], []

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

    for i, tag in enumerate(pred_tags):
      words.append(tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]))
      tags.append(tag)
    return words, tags

idx_list = list(ner_train_data.id.values)

index_list=[]
text_list= []
outputs_list=[]
tags_list =[]

for i in tqdm(range(len(content_sentence_list))):
    for text in content_sentence_list[i]:
    # text = content_sentence_list[i]
        words,tags = ner_inference(text[:510])
        index_list.append(idx_list[i])

        text_list.append(text)
        tags_list.append(tags)
        outputs_list.append(words)

content_df=pd.DataFrame(zip(index_list,text_list,outputs_list,tags_list))
content_df.columns=['id','text','output','tag']

# content_df['title'] = list(ner_train_data.title.values)
# content_df['content']=list(ner_train_data.content.values)

index_list=[]
text_list= []
outputs_list=[]
tags_list =[]

for i in tqdm(range(len(title_sentence_list))):
#     for text in title_sentence_list[i]:
    text = title_sentence_list[i]
    words,tags = ner_inference(text[:510])
    index_list.append(idx_list[i])

    text_list.append(text)
    tags_list.append(tags)
    outputs_list.append(words)

title_df=pd.DataFrame(zip(index_list,text_list,outputs_list,tags_list))
title_df.columns=['id','text','output','tag']

title_df['title'] = list(ner_train_data.title.values)
title_df['content']=list(ner_train_data.content.values)

import pandas as pd
import ast
#unique_tags

def export(test_tag, test_text, target_B_tag, target_I_tag):

    # b index list
    b_list = [i for i, x in enumerate(test_tag) if x == target_B_tag]

    # 임시 i list (나중에 b list랑 더함)
    ii_list = []
    for n in range(len(b_list)):
        if n + 1 < len(b_list):
            sample = test_tag[(b_list[n]):(b_list[n+1])]
        else:
            sample = test_tag[(b_list[n]):]

        sample.reverse()

        if target_I_tag in sample:
            ii_list.append(len(sample) - sample.index(target_I_tag))
        else:
            ii_list.append(0)


    # i list만들기 (임시 i list + b list)
    i_list = []
    for i, j in zip(b_list, ii_list):
        i_list.append(i + j)


    # b, i 매칭 -> 튜플로
    ticker_range = list(zip(b_list, i_list))

    # 티커 리스트
    ticker_list = []

    for r in ticker_range:
        ticker_list.append(test_text[r[0]-1:r[1]-1])

    return ticker_list

tag_list = [
'DAT',
'LOC',
'ORG',
'PER']

for y in tqdm(tag_list):
    content_df[y] = content_df.apply(lambda x: export(x.tag, x.text,'B-'+y,'I-'+y), axis=1)
    content_df[y]=content_df[y].apply(lambda x : ','.join(x))

for y in tqdm(tag_list):
    title_df[y] = title_df.apply(lambda x: export(x.tag, x.text,'B-'+y,'I-'+y), axis=1)
    title_df[y]=title_df[y].apply(lambda x : ','.join(x))

cont_synthetic_entities = []
cont_entities_list = []
for i in tqdm(idx_list):
    temp_entities = []
    temp_list = []

    for class_name in tag_list:
        #class_name, entity_names = real_ent['class_name'], real_ent['entity_names']

        #함수 사용
        #syn_entities = postprocess_entities(syn_entities)
        #df[df.loc[:,'news_index']==i].class_name.to_set().to_list()
        newlist = [x for x in list(content_df[content_df.loc[:,'id']==i][class_name]) if pd.isnull(x) == False]
        loc_tag_list = [item for element in newlist for item in element.split(',')]

        syn_entities = list(set(loc_tag_list))

        #syn_entities = [x for x in syn_entities if pd.isnull(x) == False]
        #syn_entities = list(set(syn_entities))
        #if len(syn_entities) != 0:
            #temp.append({class_name : syn_entities})
        temp_entities.append({class_name : syn_entities})
        temp_list.append(syn_entities)
    cont_synthetic_entities.append(temp_entities)
    cont_entities_list.append(temp_list)

title_synthetic_entities = []
title_entities_list = []
for i in tqdm(idx_list):
    temp_entities = []
    temp_list = []

    for class_name in tag_list:
        #class_name, entity_names = real_ent['class_name'], real_ent['entity_names']

        #함수 사용
        #syn_entities = postprocess_entities(syn_entities)
        #df[df.loc[:,'news_index']==i].class_name.to_set().to_list()
        newlist = [x for x in list(title_df[title_df.loc[:,'id']==i][class_name]) if pd.isnull(x) == False]
        loc_tag_list = [item for element in newlist for item in element.split(',')]

        syn_entities = list(set(loc_tag_list))

        #syn_entities = [x for x in syn_entities if pd.isnull(x) == False]
        #syn_entities = list(set(syn_entities))
        #if len(syn_entities) != 0:
            #temp.append({class_name : syn_entities})
        temp_entities.append({class_name : syn_entities})
        temp_list.append(syn_entities)
    title_synthetic_entities.append(temp_entities)
    title_entities_list.append(temp_list)

content_tag_list = ['content_DAT','content_LOC','content_ORG','content_PER']
title_tag_list = ['title_DAT','title_LOC','title_ORG','title_PER']

content_ner_df = pd.DataFrame(cont_entities_list, columns = content_tag_list)

for tag in tqdm(content_tag_list):
    content_ner_df.loc[:,tag] = content_ner_df.loc[:,tag].apply(lambda x : str(x).replace('[', '').replace(']', '').replace('\'', '')).values

#content_ner_df

title_ner_df = pd.DataFrame(title_entities_list, columns = title_tag_list)

for tag in tqdm(title_tag_list):
    title_ner_df.loc[:,tag] = title_ner_df.loc[:,tag].apply(lambda x : str(x).replace('[', '').replace(']', '').replace('\'', '')).values

final_train_ner = pd.concat([title_ner_df,content_ner_df],axis=1)

final_train_ner['id'] = list(ner_train_data.id.values)
final_train_ner['title'] = list(ner_train_data.title.values)
final_train_ner['content']=list(ner_train_data.content.values)

final_train_ner = final_train_ner.loc[:,['id','title','content','title_DAT','title_LOC','title_ORG','title_PER','content_DAT','content_LOC','content_ORG','content_PER']]

final_train_ner.to_csv('train_ner.csv')

ner_test_data = test_data.fillna('').reset_index(drop = True)
idx_list = list(ner_test_data.id.values)
test_sentence_list=[]
index_list=[]
for i in tqdm(range(ner_test_data.shape[0])): #list(kr_df.index)
# # for i in tqdm(list(kr_df.index)): #list(kr_df.index)

    s_text=splitter(ner_test_data.content[i])
    test_sentence_list.append(s_text)
    count=len(s_text)
    index_list.append(count)

# test_sentence_list = ner_test_data.content

content_sentence_list=[]
# content_sentence_list = ner_test_data.content
content_sentence_list = test_sentence_list

title_sentence_list = []
title_sentence_list = ner_test_data.title

index_list=[]
text_list= []
outputs_list=[]
tags_list =[]

for i in tqdm(range(len(content_sentence_list))):
    for text in content_sentence_list[i]:
    # text = content_sentence_list[i]
        words,tags = ner_inference(text[:510])
        index_list.append(idx_list[i])

        text_list.append(text)
        tags_list.append(tags)
        outputs_list.append(words)

content_df=pd.DataFrame(zip(index_list,text_list,outputs_list,tags_list))
content_df.columns=['id','text','output','tag']

# content_df['title'] = list(ner_test_data.title.values)
# content_df['content']=list(ner_test_data.content.values)

index_list=[]
text_list= []
outputs_list=[]
tags_list =[]

for i in tqdm(range(len(title_sentence_list))):
#     for text in sentence_list[i]:
    text = title_sentence_list[i]
    words,tags = ner_inference(text[:510])
    index_list.append(idx_list[i])

    text_list.append(text)
    tags_list.append(tags)
    outputs_list.append(words)

title_df=pd.DataFrame(zip(index_list,text_list,outputs_list,tags_list))
title_df.columns=['id','text','output','tag']

title_df['title'] = list(ner_test_data.title.values)
title_df['content']=list(ner_test_data.content.values)

for y in tqdm(tag_list):
    content_df[y] = content_df.apply(lambda x: export(x.tag, x.text,'B-'+y,'I-'+y), axis=1)
    content_df[y]=content_df[y].apply(lambda x : ','.join(x))

for y in tqdm(tag_list):
    title_df[y] = title_df.apply(lambda x: export(x.tag, x.text,'B-'+y,'I-'+y), axis=1)
    title_df[y]=title_df[y].apply(lambda x : ','.join(x))

cont_synthetic_entities = []
cont_entities_list = []
for i in tqdm(idx_list):
    temp_entities = []
    temp_list = []

    for class_name in tag_list:
        #class_name, entity_names = real_ent['class_name'], real_ent['entity_names']

        #함수 사용
        #syn_entities = postprocess_entities(syn_entities)
        #df[df.loc[:,'news_index']==i].class_name.to_set().to_list()
        newlist = [x for x in list(content_df[content_df.loc[:,'id']==i][class_name]) if pd.isnull(x) == False]
        loc_tag_list = [item for element in newlist for item in element.split(',')]

        syn_entities = list(set(loc_tag_list))

        #syn_entities = [x for x in syn_entities if pd.isnull(x) == False]
        #syn_entities = list(set(syn_entities))
        #if len(syn_entities) != 0:
            #temp.append({class_name : syn_entities})
        temp_entities.append({class_name : syn_entities})
        temp_list.append(syn_entities)
    cont_synthetic_entities.append(temp_entities)
    cont_entities_list.append(temp_list)

title_synthetic_entities = []
title_entities_list = []
for i in tqdm(idx_list):
    temp_entities = []
    temp_list = []

    for class_name in tag_list:
        #class_name, entity_names = real_ent['class_name'], real_ent['entity_names']

        #함수 사용
        #syn_entities = postprocess_entities(syn_entities)
        #df[df.loc[:,'news_index']==i].class_name.to_set().to_list()
        newlist = [x for x in list(title_df[title_df.loc[:,'id']==i][class_name]) if pd.isnull(x) == False]
        loc_tag_list = [item for element in newlist for item in element.split(',')]

        syn_entities = list(set(loc_tag_list))

        #syn_entities = [x for x in syn_entities if pd.isnull(x) == False]
        #syn_entities = list(set(syn_entities))
        #if len(syn_entities) != 0:
            #temp.append({class_name : syn_entities})
        temp_entities.append({class_name : syn_entities})
        temp_list.append(syn_entities)
    title_synthetic_entities.append(temp_entities)
    title_entities_list.append(temp_list)

content_tag_list = ['content_DAT','content_LOC','content_ORG','content_PER']
title_tag_list = ['title_DAT','title_LOC','title_ORG','title_PER']

content_ner_df = pd.DataFrame(cont_entities_list, columns = content_tag_list)

for tag in tqdm(content_tag_list):
    content_ner_df.loc[:,tag] = content_ner_df.loc[:,tag].apply(lambda x : str(x).replace('[', '').replace(']', '').replace('\'', '')).values

title_ner_df = pd.DataFrame(title_entities_list, columns = title_tag_list)

for tag in tqdm(title_tag_list):
    title_ner_df.loc[:,tag] = title_ner_df.loc[:,tag].apply(lambda x : str(x).replace('[', '').replace(']', '').replace('\'', '')).values

final_test_ner = pd.concat([title_ner_df,content_ner_df],axis=1)

final_test_ner['title'] = list(ner_test_data.title.values)
final_test_ner['id'] = list(ner_test_data.id.values)
final_test_ner['content']=list(ner_test_data.content.values)

final_test_ner = final_test_ner.loc[:,['id','title','content','title_DAT','title_LOC','title_ORG','title_PER','content_DAT','content_LOC','content_ORG','content_PER']]
final_test_ner.to_csv('test_ner.csv')









