!pip install git+https://github.com/huggingface/transformers.git
!pip install datasets
!pip install sentencepiece

!pip install tqdm

!pip install wandb

!sudo apt-get install git-lfs

import torch

from datasets import load_dataset, load_metric, list_metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast

from tqdm import tqdm

from typing import Dict, List, Optional

import dataclasses
from dataclasses import dataclass, field

import logging
import os
import sys

import numpy as np
import torch

from huggingface_hub import notebook_login

from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    EvalPrediction,
    DataCollator,
    Trainer,
    TrainingArguments)
    
import wandb
wandb.login()

%env WANDB_PROJECT=greek-m2m100-4ep-512 # different for each model

notebook_login()

!git config --global user.email "chaido.porlou@gmail.com"
!git config --global user.name "chaido13"

from datasets import load_dataset

raw_dataset = load_dataset("/content/modified_squad.py", use_auth_token=True)

###################################################################################### MODELS

# from transformers import MT5ForConditionalGeneration, T5TokenizerFast

# checkpoint = "google/mt5-base"
# model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
# tokenizer = T5TokenizerFast.from_pretrained(checkpoint)

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

checkpoint = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = M2M100Tokenizer.from_pretrained(checkpoint)

# from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

# checkpoint = "facebook/nllb-200-distilled-600M"
# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# tokenizer = NllbTokenizerFast.from_pretrained(checkpoint)

######################################################################################

tokenizer.sep_token = '<sep>'
tokenizer.add_tokens(['<sep>'])
model.resize_token_embeddings(len(tokenizer))

tokenizer.sep_token_id

max_input_length =  512
max_target_length = 64

# tokenize the examples
def convert_to_features(example_batch):

    input_encodings = tokenizer.batch_encode_plus(example_batch['context'], 
                                                  max_length=max_input_length, 
                                                  add_special_tokens=True,
                                                  truncation=True, 
                                                  pad_to_max_length=True)
    
    target_encodings = tokenizer.batch_encode_plus(example_batch['questions'], 
                                                   max_length=max_target_length, 
                                                   add_special_tokens=True,
                                                   truncation=True, 
                                                   pad_to_max_length=True)
                                                   
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids']
        ,'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings

def add_eos_examples(example):
  example['context'] = example['context'] + " </s>"
  example['questions'] = example['questions'] + " </s>"
  return example

def add_special_tokens(example):
  example['questions'] = example['questions'].replace("{sep_token}", '<sep>')
  return example

######################################################################################

tokenized_dataset  = raw_dataset.map(add_eos_examples)
tokenized_dataset = tokenized_dataset.map(add_special_tokens)
tokenized_dataset  = tokenized_dataset.map(convert_to_features,  batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(
    ["context", "questions"]
)

train_dataset = tokenized_dataset["train"]
valid_dataset = tokenized_dataset["validation"]

columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

torch.save(train_dataset, 'train_data.pt')
torch.save(valid_dataset, 'valid_data.pt')

######################################################################################

# This dataclass implementation is taken from Suraj Patil: https://github.com/patil-suraj/question_generation
@dataclass
class T2TDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
    A dictionary of tensors
    """
    
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100 
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])
    
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        'decoder_attention_mask': decoder_attention_mask
    }
    
training_args = TrainingArguments(output_dir="/content/new_model", 
                                  per_device_train_batch_size=4, 
                                  per_device_eval_batch_size=4,
                                  gradient_accumulation_steps=16,
                                  learning_rate=1e-4, 
                                  num_train_epochs=4,
                                  logging_steps=100,
                                  run_name="greek-m2m100-4ep-512",
                                  evaluation_strategy="steps",
                                  save_steps=500,
                                  report_to="wandb",
                                  push_to_hub=True,
                                  hub_model_id="chaido13/greek-m2m100-4ep-512") # different for each model
                                  

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.cuda.empty_cache()

logger = logging.getLogger(__name__)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=T2TDataCollator()
)

# Training
trainer.train()

# When training is done, we push the fine-tuned model to the Hub
# different for each model
trainer.push_to_hub("greek-m2m100-4ep-512")
tokenizer.push_to_hub("greek-m2m100-4ep-512")

wandb.finish()

###################################################################################### EXAMPLE OF USE

from transformers import MT5ForConditionalGeneration, T5TokenizerFast
# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
# from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

# can be used for all models with the correct model and tokenizer classes
hfmodel = MT5ForConditionalGeneration.from_pretrained("chaido13/greek-mt5-4ep-512")
tokenizer = T5TokenizerFast.from_pretrained("chaido13/greek-mt5-4ep-512")

def hf_run_model(input_string, **generator_args):
    generator_args = {
    "max_length": 384,
    "num_beams": 3,
    "length_penalty": 1.5,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "num_return_sequences": 1
    }

    input_string = "generate questions: " + input_string + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = hfmodel.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)  
    output = [item.split("<sep>") for item in output]
    
    return output

text = "Μετά το θάνατο του Πτολεμαίου ΙΒ΄ το 51 π.Χ., ξεκίνησε η βασιλεία της μεγαλύτερης εν ζωή θυγατέρας του, της Κλεοπάτρας Ζ΄, με την οποία έμελλε να λάβει τέλος η δυναστεία που ίδρυσε ο πολυπράγμων Μακεδόνας στρατηγός Πτολεμαίος Α΄ ο Λάγου περίπου τριακόσια χρόνια πριν. Όταν η Κλεοπάτρα ανέβηκε στον θρόνο, το βασίλειό της ήδη έμοιαζε να έχει φτάσει στη δύση του: σημαντικές επαρχίες του, η Κοίλη Συρία, η Κυρήνη, η Κύπρος, ήδη είχαν χαθεί. Επίσης, η δημοτικότητα του βασιλικού οίκου ποτέ δεν ήταν χαμηλότερη, καθώς πλέον οι Ρωμαίοι επενέβαιναν απρόσκοπτα στα εσωτερικά του κράτους, το οποίο είχε μεταλλαχθεί σε ανεπίσημη ρωμαϊκή επαρχία. Όλα έδειχναν πως οι Πτολεμαίοι θα έσβηναν άδοξα, όπως οι Σελευκίδες λίγα χρόνια νωρίτερα. Ωστόσο, η δυναστεία, λίγο πριν το τέλος έμελλε να γνωρίσει μία τελευταία αναλαμπή δόξας, αγγίζοντας το όνειρο της κατάκτησης μιας αυτοκρατορίας μεγαλύτερης εδαφικά και από το κράτος των πρώτων Πτολεμαίων. Και επειδή η στρατιωτική δύναμη του κράτους δεν μπορούσε να υπερκεράσει αυτή του ρωμαϊκού κολοσσού, η τελευταία ηγεμόνας δεν χρησιμοποίησε όπως οι πρόγονοί της τη δύναμη των όπλων, αλλά μια νέα: αυτή της προσωπικής γοητείας."

import textwrap
lines = textwrap.wrap(text, 384, break_long_words=False, fix_sentence_endings=True)

questions = []
temp_list = []

for j in range(len(lines)):
  questions.append(hf_run_model(lines[j]))

for m in range(len(questions)):
    for k in range(len(questions[m])):
        temp = questions[m][k][0].split('; ')
        for p in range(len(temp)-1):
            temp_list.append(temp[p] + ';')

qg_result = temp_list

print(qg_result)
