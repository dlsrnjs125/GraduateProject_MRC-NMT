import os
import random
import math
import csv
import json
from statistics import mean
from typing import List, Tuple, Dict, Any
import uuid

from tqdm.notebook import tqdm
from easydict import EasyDict as edict

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# import wandb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torchinfo import summary

from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def read_dev_klue(path):
    with open(path, 'rb') as f:
        klue_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in tqdm(klue_dict['data']):
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                temp_answer = []
                for answer in qa['answers']:
                    temp_answer.append(answer['text'])
                if len(temp_answer) != 0:  # answers의 길이가 0 == 답변할 수 없는 질문
                    contexts.append(context)
                    questions.append(question)
                    answers.append(temp_answer)

    return contexts, questions, answers


def prediction(contexts, questions):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.eval()

    result = []

    with torch.no_grad():
        for context, question in zip(contexts, questions):
            encodings = tokenizer(context, question, max_length=512, truncation=True,
                                  padding="max_length", return_token_type_ids=False)
            encodings = {key: torch.tensor([val])
                         for key, val in encodings.items()}

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            token_start_index, token_end_index = start_logits.argmax(
                dim=-1), end_logits.argmax(dim=-1)
            pred_ids = input_ids[0][token_start_index: token_end_index + 1]
            pred = tokenizer.decode(pred_ids)
            result.append(pred)

    return result


def em_evalutate(prediction_answers, real_answers):
    total = len(prediction_answers)
    exact_match = 0
    for prediction_answer, real_answer in zip(prediction_answers, real_answers):
        if prediction_answer in real_answer:
            exact_match += 1

    return (exact_match/total) * 100


if __name__ == '__main__':
    start_visualize = []
    end_visualize = []

    with torch.no_grad(), open(f'submissions/{args.NAME}.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['Id', 'Predicted'])

        rows = []
        c = 0
        # for sample in tqdm(test_dataset, "Testing"):
        for sample in tqdm(indexed_test_dataset, "Testing"):
            input_ids, token_type_ids = [torch.tensor(
                sample[key], dtype=torch.long, device="cuda") for key in ("input_ids", "token_type_ids")]
            # print(sample)

            model.eval()
            with torch.no_grad():
                output = model(
                    input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

            start_logits = output.start_logits
            end_logits = output.end_logits
            start_logits.squeeze_(0), end_logits.squeeze_(0)

            start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
            end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

            probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

            # 토큰 길이 8까지만
            for row in range(len(start_prob) - 8):
                probability[row] = torch.cat(
                    (probability[row][:8+row].cpu(), torch.Tensor([0] * (len(start_prob)-(8+row))).cpu()), 0)

            index = torch.argmax(probability).item()

            start = index // len(end_prob)
            end = index % len(end_prob)

            # 확률 너무 낮으면 자르기
            if start_prob[start] > 0.3 and end_prob[end] > 0.3:
                start_str = sample['position'][start][0]
                end_str = sample['position'][end][1]
            else:
                start_str = 0
                end_str = 0

            start_visualize.append(
                (list(start_prob.cpu()), (start, end), (start_str, end_str)))
            end_visualize.append(
                (list(end_prob.cpu()), (start, end), (start_str, end_str)))

            rows.append([sample["guid"], sample['context'][start_str:end_str]])

        writer.writerows(rows)

    idx = 0

    start_visualize = np.array(start_visualize)
    end_visualize = np.array(end_visualize)

    start_probalilities, token_pos, str_pos = start_visualize[:,
                                                              0], start_visualize[:, 1], start_visualize[:, 2]
    end_probalilities, token_pos, str_pos = end_visualize[:,
                                                          0], end_visualize[:, 1], end_visualize[:, 2]

    plt.plot(start_probalilities[idx], label="start probability")
    plt.plot(end_probalilities[idx], label="end probability")
    plt.xlabel("context token index")
    plt.ylabel("probablilty")
    plt.legend()
    plt.show()

    print('token position:', token_pos[idx])
    print('context position:', str_pos[idx])

    for i, (start, end) in enumerate(token_pos):
    if end - start > 1:
        if i > 0:
            plt.plot(start_probalilities[i])
            plt.plot(end_probalilities[i])
            print(i, start, end)
            break

    temp = []
    h = 0
    l = 100
    for i, (start, end) in enumerate(token_pos):
        h = max(h, end - start)
        l = min(l, end - start)
        temp.append(end - start)
    plt.plot(temp)
    print(mean(temp))

    mu = mean(temp)
    sigma = math.sqrt(np.var(temp))
    x = np.linspace(-100, 100, len(temp))
    g = (1 / np.sqrt(2*np.pi * sigma**2)) * np.exp(- (x-mu)**2 / (2*sigma**2))
    plt.title('Gaussian')
    plt.plot(x, g)

    z = [(i-mu)/sigma for i in temp]
    print(f'평균: {round(mean(z), 9)}')
    print(f'표준편차: {math.sqrt(np.var(z))}')
    print('-----90%------')
    print(mu - 1.645*sigma/math.sqrt(len(temp)))
    print(mu + 1.645*sigma/math.sqrt(len(temp)))
    print('-----95%------')
    print(mu - 1.96*sigma/math.sqrt(len(temp)))
    print(mu + 1.96*sigma/math.sqrt(len(temp)))
    print('-----99%------')
    print(mu - 2.576*sigma/math.sqrt(len(temp)))
    print(mu + 2.576*sigma/math.sqrt(len(temp)))

    dev_contexts, dev_questions, dev_answers = read_dev_klue(
        "/content/klue-mrc-v1.1_dev.json")
    pred_answers = prediction(dev_contexts, dev_questions)
    print(pred_answers)
    em_score = em_evalutate(pred_answers, dev_answers)
    print(em_score)


class exact_match():
    def __init__(self, model, tokenizer, path) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.path = path
        self.contexts, self.questions, self.answers = read_dev_klue(path)

    def read_dev_klue(self, path):
        with open(path, 'rb') as f:
            klue_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in tqdm(klue_dict['data']):
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    temp_answer = []
                    for answer in qa['answers']:
                        temp_answer.append(answer['text'])
                    if len(temp_answer) != 0:  # answers의 길이가 0 == 답변할 수 없는 질문
                        contexts.append(context)
                        questions.append(question)
                        answers.append(temp_answer)

        return contexts, questions, answers

    def prediction(self, contexts, questions):
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(device)
        self.model.eval()

        result = []

        with torch.no_grad():
            for context, question in zip(contexts, questions):
                encodings = self.tokenizer(context, question, max_length=512, truncation=True,
                                           padding="max_length", return_token_type_ids=False)
                encodings = {key: torch.tensor([val])
                             for key, val in encodings.items()}

                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                token_start_index, token_end_index = start_logits.argmax(
                    dim=-1), end_logits.argmax(dim=-1)
                pred_ids = input_ids[0][token_start_index: token_end_index + 1]
                pred = self.tokenizer.decode(pred_ids)
                result.append(pred)

        return result

    def em_evalutate(self, prediction_answers, real_answers):
        total = len(prediction_answers)
        exact_match = 0
        for prediction_answer, real_answer in zip(prediction_answers, real_answers):
            if prediction_answer in real_answer:
                exact_match += 1

        return (exact_match/total) * 100


class levenshtein():
    def __init__(self, s1, s2) -> None:
        self.s1 = s1
        self.s2 = s2

    def levenshtein_score(self):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i+1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j+1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
