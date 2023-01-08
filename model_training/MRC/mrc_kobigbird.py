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

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torchinfo import summary

from transformers import  AutoModelForQuestionAnswering, AutoTokenizer

import gc

args = edict({'w_project': 'test_project',
              'w_entity': 'chohs1221',
              'learning_rate': 5e-5,
              'batch_size': {'train': 256,
                             'eval': 16,
                             'test': 256},
              'accumulate': 64,
              'epochs': 30,
              'seed': 42,
              # 'model_name': 'monologg/koelectra-base-v3-discriminator',
              'model_name': 'monologg/kobigbird-bert-base',
              'max_length': 2048})
args['NAME'] = ''f'kobigbird_ep{args.epochs}_max{args.max_length}_lr{args.learning_rate}_{random.randrange(0, 1024)}'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

class KoMRC:
    def __init__(self, data, indices: List[Tuple[int, int, int]]):
        self._data = data
        self._indices = indices


    # Json을 불러오는 메소드
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as fd:
            data = json.load(fd)

        indices = []
        for d_id, document in enumerate(data['data']):
            for p_id, paragraph in enumerate(document['paragraphs']):
                for q_id, _ in enumerate(paragraph['qas']):
                    indices.append((d_id, p_id, q_id))

        return cls(data, indices)


    # 데이터 셋을 잘라내는 메소드
    @classmethod
    def split(cls, dataset, eval_ratio: float=.1):
        indices = list(dataset._indices)
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * eval_ratio):]
        eval_indices = indices[:int(len(indices) * eval_ratio)]

        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data['data'][d_id]['paragraphs'][p_id]

        qa = paragraph['qas'][q_id]

        guid = qa['guid']

        context = paragraph['context'].replace('\n', 'n').replace('\xad', '')

        question = qa['question'].replace('\n', 'n').replace('\xad', '')

        answers = qa['answers']
        if answers != None:
            for a in answers:
                a['text'] = a['text'].replace('\n', 'n').replace('\xad', '')


        return {'guid': guid,
            'context': context,
            'question': question,
            'answers': answers
        }

    def __len__(self) -> int:
        return len(self._indices)
class TokenizedKoMRC(KoMRC):
    def __init__(self, data, indices: List[Tuple[int, int, int]]) -> None:
        super().__init__(data, indices)
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
        sample = super().__getitem__(index)
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

class Indexer:
    def __init__(self, vocabs: List[str], max_length: int=args.max_length):
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

class Collator:
    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer


    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = {key: [sample[key] for sample in samples] for key in samples[0]}

        for key in 'start', 'end':
            if samples[key][0] is None:
                samples[key] = None
            else:
                samples[key] = torch.tensor(samples[key], dtype=torch.long)

        for key in 'input_ids', 'attention_mask', 'token_type_ids':
            samples[key] = pad_sequence([torch.tensor(sample, dtype=torch.long) for sample in samples[key]],
                                        batch_first=True,
                                        padding_value=self._indexer.pad_id)

        return samples
if __name__ == '__main__':
    for name in 'models', 'submissions':
        os.makedirs(name, exist_ok=True)

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # preprocessing

    # dataset = KoMRC.load('/content/train.json')
    # train_dataset, dev_dataset = KoMRC.split(dataset)
    dataset = TokenizedKoMRC.load('/content/train.json')
    train_dataset, dev_dataset = TokenizedKoMRC.split(dataset)
    indexer = Indexer(list(tokenizer.vocab.keys()))

    indexed_train_dataset = IndexerWrappedDataset(train_dataset, indexer)
    indexed_dev_dataset = IndexerWrappedDataset(dev_dataset, indexer)

    sample = indexed_dev_dataset[0]

    collator = Collator(indexer)
    train_loader = DataLoader(indexed_train_dataset,
                              batch_size = args.batch_size.train // args.accumulate,
                              shuffle = True,
                              collate_fn = collator,
                              num_workers = 2)

    dev_loader = DataLoader(indexed_dev_dataset,
                            batch_size = args.batch_size.eval,
                            shuffle = False,
                            collate_fn = collator,
                            num_workers = 2)
    batch = next(iter(dev_loader))

    gc.collect()
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # training
    train_losses = []
    dev_losses = []

    train_loss = []
    dev_loss = []

    loss_accumulate = 0.

    best_model = [-1, int(1e9)]

    for epoch in range(args.epochs):
        print("Epoch", epoch, '===============================================================================================================')

        # Train
        progress_bar_train = tqdm(train_loader, desc='Train')
        for i, batch in enumerate(progress_bar_train, 1):
            del batch['guid'], batch['context'], batch['question'], batch['position']
            batch = {key: value.cuda() for key, value in batch.items()}

            start = batch.pop('start')
            end = batch.pop('end')

            output = model(**batch)

            start_logits = output.start_logits
            end_logits = output.end_logits

            loss = (F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)) / args.accumulate
            loss.backward()

            loss_accumulate += loss.item()

            del batch, start, end, start_logits, end_logits, loss

            if i % args.accumulate == 0:
                # clip_grad_norm_(model.parameters(), max_norm=1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=False)

                train_loss.append(loss_accumulate)
                progress_bar_train.set_description(f"Train - Loss: {loss_accumulate:.3f}")
                loss_accumulate = 0.
            else:
                continue

            if i % int(len(train_loader) / (args.accumulate * 25)) == 0:
                # Evaluation
                for batch in dev_loader:
                    del batch['guid'], batch['context'], batch['question'], batch['position']
                    batch = {key: value.cuda() for key, value in batch.items()}

                    start = batch.pop('start')
                    end = batch.pop('end')

                    model.eval()
                    with torch.no_grad():
                        output = model(**batch)

                        start_logits = output.start_logits
                        end_logits = output.end_logits
                    model.train()

                    loss = F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)

                    dev_loss.append(loss.item())

                    del batch, start, end, start_logits, end_logits, loss

                train_losses.append(mean(train_loss))
                dev_losses.append(mean(dev_loss))
                train_loss = []
                dev_loss = []


                if dev_losses[-1] <= best_model[1]:
                    best_model = (epoch, dev_losses[-1])
                    model.save_pretrained(f'models/{args.NAME}_{epoch}')
                    # print(f'model saved!!\nvalid_loss: {dev_losses[-1]}')

                # wandb.log({"train_loss": train_losses[-1],
                          #  "valid_loss": dev_losses[-1]})


        print(f"Train Loss: {train_losses[-1]:.3f}")
        print(f"Valid Loss: {dev_losses[-1]:.3f}")
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    # test
    model = AutoModelForQuestionAnswering.from_pretrained(f'models/{args.NAME}_{best_model[0]}')
    model.cuda();

    for idx, sample in zip(range(1, 4), indexed_train_dataset):
        print(f'------{idx}------')
        print('Context:', sample['context'])
        print('Question:', sample['question'])
        
        input_ids, token_type_ids = [
            torch.tensor(sample[key], dtype=torch.long, device="cuda")
            for key in ("input_ids", "token_type_ids")
        ]

        model.eval()
        with torch.no_grad():
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

        print('Answer:', sample['context'][start_str:end_str])
