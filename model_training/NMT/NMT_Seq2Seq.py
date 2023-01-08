#%%
from typing import Dict, List
import csv

from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer as BaseGPT2Tokenizer,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer
)

import torch

from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel

from lib.tokenization_kobert import KoBertTokenizer

#%%
class PairedDataset:
    def __init__(self, 
        src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer,
        file_path: str
    ):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = tgt_tokenizer
        with open(file_path, 'r') as fd:
            self.data = [row[1:] for row in csv.reader(fd)]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src, trg = self.data[index]
        embeddings = self.src_tokenizer(src, return_attention_mask=False, return_token_type_ids=False)
        embeddings['labels'] = self.trg_tokenizer(trg, return_attention_mask=False)['input_ids']

        return embeddings

    def __len__(self):
        return len(self.data)

#%%
class GPT2Tokenizer(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]

src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
trg_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
dataset = PairedDataset(src_tokenizer, trg_tokenizer, 'data/train.csv')
eval_dataset = PairedDataset(src_tokenizer, trg_tokenizer, 'data/dev.csv')

# %%
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'monologg/distilkobert',
    'distilgpt2',
    pad_token_id=trg_tokenizer.bos_token_id
)
model.config.decoder_start_token_id = trg_tokenizer.bos_token_id

# %%
collator = DataCollatorForSeq2Seq(src_tokenizer, model)

arguments = Seq2SeqTrainingArguments(
    output_dir='dump',
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=30,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1,
    save_total_limit=5,
    dataloader_num_workers=1,
    fp16=True,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model,
    arguments,
    data_collator=collator,
    train_dataset=dataset,
    eval_dataset=eval_dataset
)
# %%
trainer.train()

model.save_pretrained("dump/best_model")
# %%
