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

train_ner_path = '/content/drive/MyDrive/nlp_data/file2/train_ner.csv'
test_ner_path = '/content/drive/MyDrive/nlp_data/file2/test_ner.csv'

train_data = pd.read_csv(train_ner_path)
test_data = pd.read_csv(test_ner_path)

for i, d in enumerate(train_data['title']):
    if pd.notnull(train_data['title_LOC'][i]) and pd.notnull(d):
        title_loc_elements = str(train_data['title_LOC'][i]).split(', ')
        for element in title_loc_elements:
            if element.strip() in str(d):
                tok = "[LOC]" + element.strip() + "[LOC]"
                train_data.loc[i, 'title'] = str(d).replace(element.strip(), tok)
for i, d in enumerate(train_data['title']):
    if pd.notnull(train_data['title_DAT'][i]) and pd.notnull(d):
        title_loc_elements = str(train_data['title_DAT'][i]).split(', ')
        for element in title_loc_elements:
            if element.strip() in str(d):
                tok = "[DAT]" + element.strip() + "[DAT]"
                train_data.loc[i, 'title'] = str(d).replace(element.strip(), tok)
for i, d in enumerate(train_data['title']):
    if pd.notnull(train_data['title_PER'][i]) and pd.notnull(d):
        title_loc_elements = str(train_data['title_PER'][i]).split(', ')
        for element in title_loc_elements:
            if element.strip() in str(d):
                tok = "[PER]" + element.strip() + "[PER]"
                train_data.loc[i, 'title'] = str(d).replace(element.strip(), tok)

for i, d in enumerate(train_data['content']):
    if pd.notnull(train_data['content_LOC'][i]) and pd.notnull(d):
        content_loc_elements = str(train_data['content_LOC'][i]).split(', ')
        for element in content_loc_elements:
            if element.strip() in str(d):
                tok = "[LOC]" + element.strip() + "[LOC]"
                train_data.loc[i, 'content'] = str(d).replace(element.strip(), tok)
for i, d in enumerate(train_data['content']):
    if pd.notnull(train_data['content_DAT'][i]) and pd.notnull(d):
        content_loc_elements = str(train_data['content_DAT'][i]).split(', ')
        for element in content_loc_elements:
            if element.strip() in str(d):
                tok = "[DAT]" + element.strip() + "[DAT]"
                train_data.loc[i, 'content'] = str(d).replace(element.strip(), tok)
for i, d in enumerate(train_data['content']):
    if pd.notnull(train_data['content_PER'][i]) and pd.notnull(d):
        content_loc_elements = str(train_data['content_PER'][i]).split(', ')
        for element in content_loc_elements:
            if element.strip() in str(d):
                tok = "[PER]" + element.strip() + "[PER]"
                train_data.loc[i, 'content'] = str(d).replace(element.strip(), tok)

    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

    special_tokens = ["[PER]", "[LOC]", "[DAT]"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    train_source = []
    train_target = []

    for idx, data in enumerate(train_data['content']):
        train_source.append(data)
    for idx, data in enumerate(train_data['title']):
        train_target.append(data)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    class NewsDataset(Dataset):
        def __init__(self, source, target, tokenizer, max_length=512, max_target_length=128):
            self.source = source
            self.target = target
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.max_target_length = max_target_length

        def __len__(self):
            return len(self.source)

        def __getitem__(self, idx):
            input_text = self.source[idx]
            target_text = self.target[idx]
            input_encoding = self.tokenizer(input_text, truncation=True, padding='max_length',
                                            max_length=self.max_length, return_tensors='pt')
            target_encoding = self.tokenizer(target_text, truncation=True, padding='max_length',
                                             max_length=self.max_target_length, return_tensors='pt')

            return {
                'input_ids': input_encoding['input_ids'].flatten(),
                'attention_mask': input_encoding['attention_mask'].flatten(),
                'labels': target_encoding['input_ids'].flatten(),
            }


    dataset = NewsDataset(train_source, train_target, tokenizer)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    invalid_list = []
    for i, d in enumerate(train_target):
        if (type(d) != str):
            invalid_list.append(i)
    for i in invalid_list:
        new = "NaN"
        train_target[i] = new

    model.to(device)

    for epoch in range(10):
        tqdm_dataloader = tqdm(data_loader, desc=f"epoch {epoch + 1}/{10}", unit="batch")
        try:
            for batch in tqdm_dataloader:
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    tqdm_dataloader.set_postfix({'loss': loss.item()})
                except ValueError as e:
                    continue
        except ValueError as e:
            continue

    output_dir = '/content/drive/MyDrive/fifinal'
    model.save_pretrained(output_dir)

#inference
model = BartForConditionalGeneration.from_pretrained(output_dir)

input_text = """
초등학교 교실에서 수업 중인 교사의 목을 졸랐다가 실형을 선고받은 30대 학부모가 징역 1년을 선고한 1심 판결에 불복해 항소하자
검찰도 맞항소 했습니다. A 씨는 2021년 11월 인천 한 초등학교 교실에서 수업하던 여성 교사 B 씨의 목을 조르고 팔을 강제로 끌어당겨
다치게 한 혐의로 불구속 기소됐는데요. 그는 아들이 학교폭력 가해자로 지목돼 심의위원회에 회부된다는 통보를 받자 교실에 들어가 "넌 교사 자질도 없다"라거나
"경찰·교육청과 교육부 장관에게도 이야기하겠다"며 욕설을 한 것으로 조사됐습니다.
"""

inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
input_ids = inputs['input_ids'].to('cpu')

summary_ids = model.generate(input_ids, num_beams=4, max_length=150, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
