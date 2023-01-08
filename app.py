from flask import Flask, render_template, url_for, request, redirect
import torch
import json
import os
import random
import math
import csv
import json
from statistics import mean
from typing import List, Tuple, Dict, Any
import uuid

from datetime import datetime, timedelta
import pandas as pd
news_df = pd.read_csv('news.csv',sep = '\t')
# news_df.drop(columns = ['Unnamed: 0'],inplace=True)
temp_df = news_df.copy()
def load_news(idx):
    # df = pd.read_csv(filename, sep = '\t')
    news_info = temp_df.iloc[idx]
    return news_info


def date_range(date):
    start = date.split()[0].replace('/','-')
    end = date.split()[-1].replace('/','-')
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end-start).days+1)]
    return dates

app = Flask(__name__)

from threading import Semaphore

from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

from lib.tokenization_kobert import KoBertTokenizer

class TokenizedKoMRC:
    def __init__(self, data,ori_data) -> None:
        # super().__init__(data, indices)
        self.data = data
        self.ori_data = ori_data
        self.indices = [(0,0,0)]
        self._tokenizer = tokenizer


    def _tokenize_with_position(self, sentence: str) -> List[Tuple[str, Tuple[int, int]]]:
        position = 0
        tokens = []

        sentence_tokens = []
        for word in sentence.split():
            if '[UNK]' in tokenizer.tokenize(word):
                sentence_tokens.append(word)
            else:
                sentence_tokens += tokenizer.tokenize(word)

        for morph in sentence_tokens:
            if len(morph) > 2:
                if morph[:2] == '##':
                    morph = morph[2:]

            position = sentence.find(morph, position)
            tokens.append((morph, (position, position + len(morph))))
            position += len(morph)

        return tokens


    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.ori_data
        # sample = self.data
        # sample = {'guid': guid, 'context': context, 'question': question, 'answers': answers}

        context, position = zip(*self._tokenize_with_position(sample['context']))
        context, position = list(context), list(position)

        question = self._tokenizer.tokenize(sample['question'])

        if sample['answers'] is not None:
            answers = []
            for answer in sample['answers']:
                for start, (position_start, position_end) in enumerate(position):
                    if position_start <= answer['answer_start'] < position_end:
                        break
                else:
                    print(context, answer)
                    print(answer['guid'])
                    print(answer['answer_start'])
                    raise ValueError("No mathced start position")

                target = ''.join(answer['text'].split(' '))
                source = ''
                for end, morph in enumerate(context[start:], start):
                    source += morph
                    if target in source:
                        break
                else:
                    print(context, answer)
                    print(answer['guid'])
                    print(answer['answer_start'])
                    raise ValueError("No Matched end position")

                answers.append({'start': start, 'end': end})

        else:
            answers = None

        return {
            'guid': sample['guid'],
            'context_original': sample['context'],
            'context_position': position,
            'question_original': sample['question'],
            'context': context,
            'question': question,
            'answers': answers
        }
    def __len__(self) -> int:
        return len(self.indices)
class Indexer:
    def __init__(self, vocabs: List[str], max_length: int = 2048):
        self.max_length = max_length
        self.vocabs = vocabs

    @property
    def vocab_size(self):
        return len(self.vocabs)
    @property
    def pad_id(self):
        return tokenizer.vocab['[PAD]']
    @property
    def unk_id(self):
        return tokenizer.vocab['[UNK]']
    @property
    def cls_id(self):
        return tokenizer.vocab['[CLS]']
    @property
    def sep_id(self):
        return tokenizer.vocab['[SEP]']


    def sample2ids(self, sample: Dict[str, Any],) -> Dict[str, Any]:
        context = [tokenizer.convert_tokens_to_ids(token) for token in sample['context']]
        question = [tokenizer.convert_tokens_to_ids(token) for token in sample['question']]

        context = context[:self.max_length-len(question)-3]             # Truncate context

        input_ids = [self.cls_id] + question + [self.sep_id] + context + [self.sep_id]
        token_type_ids = [0] * (len(question) + 1) + [1] * (len(context) + 2)

        if sample['answers'] is not None:
            answer = sample['answers'][0]
            start = min(len(question) + 2 + answer['start'], self.max_length - 1)
            end = min(len(question) + 2 + answer['end'], self.max_length - 1)
        else:
            start = None
            end = None

        return {
            'guid': sample['guid'],
            'context': sample['context_original'],
            'question': sample['question_original'],
            'position': sample['context_position'],
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'start': start,
            'end': end
        }
class IndexerWrappedDataset:
    def __init__(self, dataset: TokenizedKoMRC, indexer: Indexer) -> None:
        self._dataset = dataset
        self._indexer = indexer

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._indexer.sample2ids(self._dataset[index])
        sample['attention_mask'] = [1] * len(sample['input_ids'])

        return sample

def mrc(con,qu,data):
    sp = json.dumps(data)
    dataset = TokenizedKoMRC(sp,data)
    indexer = Indexer(list(tokenizer.vocab.keys()))
    indexed_real_dataset = IndexerWrappedDataset(dataset, indexer)
    sample = indexed_real_dataset[0]
    input_ids, token_type_ids = [
        torch.tensor(sample[key], dtype=torch.long, device="cuda")
        for key in ("input_ids", "token_type_ids")
    ]


    output = model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

    start_logits = output.start_logits
    end_logits = output.end_logits
    start_logits.squeeze_(0), end_logits.squeeze_(0)

    start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
    end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

    probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

    index = torch.argmax(probability).item()

    start = index // len(end_prob)
    end = index % len(end_prob)

    start_str = sample['position'][start][0]
    end_str = sample['position'][end][1]
    answer = sample['context'][start_str:end_str]
    return answer


src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
trg_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')

model_nmt = EncoderDecoderModel.from_pretrained('dump/best_model')
model_nmt.config.decoder_start_token_id = trg_tokenizer.bos_token_id
model_nmt.eval()
model_nmt.cuda()

model = AutoModelForQuestionAnswering.from_pretrained('dump/best_model_mrc')
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.eval()
model.cuda()
semaphore = Semaphore(5)

@app.route('/')
def index():
        return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

# daterange : 달력의 날짜 data변수 -> 2022/01/01 - 2022/01/01 이런식으로 데이터 들어가있음
@app.route('/newsqna', methods=['POST', 'GET'])
def newsqna():
    global temp_df
    if request.method == 'POST':
        temp_df = pd.DataFrame(columns = news_df.columns)
        daterange = request.form['daterange']
        dataRange_list = date_range(daterange)
        news_list = []
        for i in dataRange_list:
            df = news_df[news_df['datetime'] == i]
            temp_df = pd.concat([temp_df,df])
            news_list += df.values.tolist()

        # print(temp_df)


        return render_template('newsqna.html', daterange=daterange, news_list=news_list)
    else:
        return render_template('newsqna.html')


# test1 : 입력받은 뉴스 본문 데이터
# test2 : 입력받은 뉴스 질문 데이터
# 답안 데이터는 여기 함수 안에다가 아무 변수나 만들고 모델 돌릴다음에 그 변수에 저장해서 출력하면 가능!!
@app.route('/qna', methods=['POST', 'GET'])
def qna():
    if request.method == 'POST':
        test1 = request.form['test1']
        test2 = request.form['test2']
        with semaphore:
            data = {'guid': '0', 'context': test1, 'question': test2, 'answers': None}
            answer = mrc(test1,test2,data)
        return render_template('qna.html', test1=test1, test2=test2,answer = answer)
    else:
        return render_template('qna.html')

@app.route("/qna_news/<int:index>/")
def qna_news(index):
    news_info = load_news(index)
    content = news_info['content']
    return render_template('qna_news.html', content=content)

# test3 : 입력받은 뉴스 번역 데이터
# 번역된 데이터또한 위에처럼 아무 변수 만들어서 모델 돌린다음 그 변수에 저장해서 출력하면 가능!!
@app.route('/translation', methods=['POST', 'GET'])
def translation():
    if request.method == 'POST':
        test3 = request.form['test3']
        embeddings = src_tokenizer(test3, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        with semaphore:
            embeddings = {k: v.cuda() for k, v in embeddings.items()}
            output = model_nmt.generate(**embeddings)[0, 1:-1].cpu()
            del embeddings
        # return trg_tokenizer.decode(output)
        answer = trg_tokenizer.decode(output)

        return render_template('translation.html', test3=test3, answer=answer)
    else:
        return render_template('translation.html')



if __name__ == "__main__":
    app.run(debug=True)
