! pip install farm-haystack
! pip install git+https://github.com/huggingface/transformers.git

################################################################################################### QUESTION GENERATOR NODE CODE CHANGE REGARDING GENERATE FUNCTION

import logging
from typing import List, Union, Optional, Iterator
import itertools
import torch

from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, M2M100Tokenizer, NllbTokenizerFast

from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.base import BaseComponent
from haystack.nodes.preprocessor import PreProcessor
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)


class QuestionGenerator(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "valhalla/t5-base-e2e-qg",
        model_version: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 256,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.5,
        early_stopping: bool = True,
        split_length: int = 50,
        split_overlap: int = 10,
        use_gpu: bool = True,
        prompt: str = "generate questions:",
        num_queries_per_doc: int = 1,
        sep_token: str = "<sep>",
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        super().__init__()
        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        self.model.to(str(self.devices[0]))
        #################################################################################################
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        #################################################################################################
        self.num_beams = num_beams
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.preprocessor = PreProcessor()
        self.prompt = prompt
        self.num_queries_per_doc = min(num_queries_per_doc, 3)
        self.batch_size = batch_size
        self.sep_token = sep_token
        self.sep_token = self.tokenizer.sep_token or sep_token
        self.progress_bar = progress_bar

    def run(self, documents: List[Document]):  # type: ignore
        generated_questions = []
        for d in documents:
            questions = self.generate(d.content)
            curr_dict = {"document_id": d.id, "document_sample": d.content[:200], "questions": questions}
            generated_questions.append(curr_dict)
        output = {"generated_questions": generated_questions, "documents": documents}
        return output, "output_1"

    def run_batch(self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None):  # type: ignore
        generated_questions = []
        if isinstance(documents[0], Document):
            questions = self.generate_batch(
                texts=[d.content for d in documents if isinstance(d, Document)], batch_size=batch_size
            )
            questions_iterator = questions  # type: ignore
            documents_iterator = documents
        else:
            questions = self.generate_batch(
                texts=[[d.content for d in doc_list] for doc_list in documents if isinstance(doc_list, list)],
                batch_size=batch_size,
            )
            questions_iterator = itertools.chain.from_iterable(questions)  # type: ignore
            documents_iterator = itertools.chain.from_iterable(documents)  # type: ignore
        for cur_questions, doc in zip(questions_iterator, documents_iterator):
            if not isinstance(doc, Document):
                raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
            curr_dict = {"document_id": doc.id, "document_sample": doc.content[:200], "questions": cur_questions}
            generated_questions.append(curr_dict)
        output = {"generated_questions": generated_questions, "documents": documents}
        return output, "output_1"

    ################
    # CHANGES HERE #
    ################
    
    def generate(self, text: str, split_overlap: int, split_length: int) -> List[str]:
        # Performing splitting because T5 has a max input length
        # Also currently, it seems that it only generates about 3 questions for the beginning section of text
        split_texts_docs = self.preprocessor.split(
            document={"content": text},
            split_by="word",
            split_respect_sentence_boundary=False,
            split_overlap=split_overlap,
            split_length=split_length
        )
        split_texts = [
            f"{self.prompt} {text.content}" if self.prompt not in text.content else text.content
            for text in split_texts_docs
        ]

        tokenized = self.tokenizer(split_texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(self.devices[0])
        
        tokens_output = self.model.generate(
            input_ids=input_ids,
            num_beams=self.num_beams,
            max_length=self.max_length,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
            num_return_sequences=self.num_queries_per_doc,
        )

        string_output = self.tokenizer.batch_decode(tokens_output, skip_special_tokens=False)

        ret = []
        
        for split in string_output:
            output = str(split)
            output = output.replace(" +4883270", "<sep>")
            output = output.rsplit(";", 1)
            output = output[0].rsplit(".", 1)
            output = output[0] + ";" 
            output = output.replace("__en__", "")
            output = output.replace("<pad> ", "")
            output = output.replace("<pad>", "")
            output = output.replace("</s> ", "")
            output = output.replace("</s>", "")
            output = output.split(sep=" <sep> ")
            ret.extend(output)  

        return ret

    def generate_batch(
        self, texts: Union[List[str], List[List[str]]], batch_size: Optional[int] = None
    ) -> Union[List[List[str]], List[List[List[str]]]]:
        """
        Generates questions for a list of strings or a list of lists of strings.

        :param texts: List of str or list of list of str.
        :param batch_size: Number of texts to process at a time.
        """

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(texts[0], str):
            single_doc_list = True
            number_of_docs = [1 for text_list in texts]
            text_iterator = texts
        else:
            single_doc_list = False
            number_of_docs = [len(text_list) for text_list in texts]
            text_iterator = itertools.chain.from_iterable(texts)  # type: ignore

        split_texts_docs = [
            self.preprocessor.split(
                document={"content": text},
                split_by="word",
                split_respect_sentence_boundary=False,
                split_overlap=self.split_overlap,
                split_length=self.split_length,
            )
            for text in text_iterator
        ]
        split_texts = [[doc.content for doc in split if isinstance(doc.content, str)] for split in split_texts_docs]
        number_of_splits = [len(split) for split in split_texts]
        flat_split_texts = [
            f"{self.prompt} {text}" if self.prompt not in text else text
            for text in itertools.chain.from_iterable(split_texts)
        ]

        batches = self._get_batches(flat_split_texts, batch_size=batch_size)
        all_string_outputs = []
        pb = tqdm(total=len(flat_split_texts), disable=not self.progress_bar, desc="Generating questions")
        for batch in batches:
            tokenized = self.tokenizer(batch, return_tensors="pt", padding=True)
            input_ids = tokenized["input_ids"].to(self.devices[0])
            # Necessary if padding is enabled so the model won't attend pad tokens
            attention_mask = tokenized["attention_mask"].to(self.devices[0])
            tokens_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.num_beams,
                max_length=self.max_length,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                length_penalty=self.length_penalty,
                early_stopping=self.early_stopping,
                num_return_sequences=self.num_queries_per_doc,
            )

            string_output = self.tokenizer.batch_decode(tokens_output, skip_special_tokens=True)
            all_string_outputs.extend(string_output)
            pb.update(len(batch))
        pb.close()
        # Group predictions together by split
        grouped_predictions_split = []
        left_idx = 0
        right_idx = 0
        for number in number_of_splits:
            right_idx = left_idx + number * self.num_queries_per_doc
            grouped_predictions_split.append(all_string_outputs[left_idx:right_idx])
            left_idx = right_idx
        # Group predictions together by doc list
        grouped_predictions_doc_list = []
        left_idx = 0
        right_idx = 0
        for number in number_of_docs:
            right_idx = left_idx + number
            grouped_predictions_doc_list.append(grouped_predictions_split[left_idx:right_idx])
            left_idx = right_idx

        results = []
        for group in grouped_predictions_doc_list:
            group_preds = []
            for doc in group:
                doc_preds = []
                for split in doc:
                    for question in split.split(self.sep_token):
                        question = question.strip()
                        if question and question not in doc_preds:
                            doc_preds.append(question)
                group_preds.append(doc_preds)
            if single_doc_list:
                results.append(group_preds[0])
            else:
                results.append(group_preds)

        return results

    @staticmethod
    def _get_batches(texts: List[str], batch_size: Optional[int]) -> Iterator[List[str]]:
        if batch_size is None:
            yield texts
            return
        else:
            for index in range(0, len(texts), batch_size):
                yield texts[index : index + batch_size]

###################################################################################################

# BEGINNING OF EVALUATION

import re
from datasets import load_dataset

raw_dataset = load_dataset("/content/modified_squad.py")

print(raw_dataset["validation"][0])

context_list = []
questions_list = []

for i in range(len(raw_dataset["validation"])):

    context = str(raw_dataset["validation"][i]['context'])
    true_context = re.search('generate questions:(.*)', context) 
    context_list.append(true_context.group(1))

    questions = str(raw_dataset["validation"][i]['questions'])
    questions_list.append(questions.replace("{sep_token}", '<sep>'))

import os
from subprocess import Popen, PIPE, STDOUT

from pprint import pprint
from tqdm import tqdm
from haystack.nodes import FARMReader
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import QuestionAnswerGenerationPipeline, QuestionGenerationPipeline
from haystack.utils import launch_es, print_questions, print_answers
import re
import math

import logging
import random

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

question_generator = QuestionGenerator(model_name_or_path = "chaido13/greek-m2m100-4ep-512", num_queries_per_doc=1, use_gpu=True)

qg_result = []

###################################
# for validation part of squad_el #

for i in range(len(context_list)//2):
    
    if(i%100 == 0):
        print(i)
    
    question_num = questions_list[i].count("<sep>")
    split_length = (len(re.findall(r'\w+', context_list[i]))) // (question_num)
    
    if split_length == 0:
        qg_result.append("No questions.")      
    else:
        split_length_final = split_length
        qg_result.append(question_generator.generate(text = context_list[i], 
                                    split_length = split_length_final, split_overlap = split_length//2))

with open('hyp_greek.txt', 'w') as f:
    for i in range(len(qg_result)):
        if len(qg_result[i]) > questions_list[i].count('<sep>'):
            sample = random.sample(qg_result[i], questions_list[i].count('<sep>'))
            for j in range(len(sample)):        
                f.write(sample[j] + ' <sep> ')    
        else:
            for j in range(len(qg_result[i])):
                f.write(qg_result[i][j] + ' <sep> ')
        f.write('\n')                   

#####################
# for QA_eval_Greek #

# for i in range(len(context_list)):
    
#     if((i+1)%10 == 0):
#         print(i+1)

#     # number of words --> we choose this in order to make only 3 questions (sometimes 1 or 2)
#     all_words = len(re.findall(r'\w+', context_list[i]))
#     qg_result.append(question_generator.generate(text = context_list[i], 
#                                     split_length = all_words//4, split_overlap = 0))


# for k in range(5):
#     with open('hyp_greek_' + str(k) + '.txt', 'w') as f:
#         for i in range(len(qg_result)):   
#             if len(qg_result[i]) == 1:
#                 for j in range(1):
#                     f.write(qg_result[i][j] + ' <sep> ')
#             elif len(qg_result[i]) == 2:
#                 for j in range(2):
#                     f.write(qg_result[i][j] + ' <sep> ')
#             else:   
#                 sample = random.sample(qg_result[i], 3)
#                 for j in range(3):        
#                     f.write(sample[j] + ' <sep> ')    
#                     # f.write(qg_result[i][j] + ' <sep> ')              
#             f.write('\n')

with open('ref_greek.txt', 'w') as f:
    for i in range(len(context_list)//2):
        f.write(questions_list[i] + '\n')

# nlg-eval EVALUATION PACKAGE

!pip install git+https://github.com/Maluuba/nlg-eval.git@master
!nlg-eval --setup

!pip install --upgrade numpy==1.20.3

!nlg-eval --hypothesis=/content/hyp_greek.txt --references=/content/ref_greek.txt