! pip install farm-haystack

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import time

import os
from subprocess import Popen, PIPE, STDOUT

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes import PreProcessor
from haystack.nodes import FARMReader
from haystack import Pipeline
from haystack.schema import EvaluationResult, MultiLabel

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
! tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
! chown -R daemon:daemon elasticsearch-7.9.2
    
es_server = Popen(
    ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon
)    

time.sleep(40)

from haystack.utils import fetch_archive_from_http

doc_index = "docs"
label_index = "labels"


preprocessor = PreProcessor(
    split_by="word",
    split_length=384,
    split_overlap=0,
    split_respect_sentence_boundary=False,
    clean_empty_lines=False,
    clean_whitespace=False,
)

host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host=host,
    username="",
    password="",
    index=doc_index,
    label_index=label_index,
    embedding_field="emb",
    embedding_dim=768,
    excluded_meta_data=["emb"],
)

document_store.delete_documents(index=doc_index)
document_store.delete_documents(index=label_index)

document_store.add_eval_data(
    filename="/content/Datasets/dev-v2.0.json", # filename="/content/Datasets/squad_el_dev.json"
    doc_index=doc_index,
    label_index=label_index,
    preprocessor=preprocessor,
)

# English and Greek reader models
reader = FARMReader(model_name_or_path="ahotrod/albert_xxlargev1_squad2_512", top_k = 1, max_seq_len = 384, 
                    doc_stride = 128, context_window_size = 256, return_no_answer=True)    
# reader = FARMReader(model_name_or_path="Danastos/squad_bert_el", top_k = 1, max_seq_len = 384, doc_stride = 128, 
#                     context_window_size = 256, return_no_answer=True, use_auth_token = "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")    

pipe = Pipeline()
pipe.add_node(component=reader, name='Reader', inputs=['Query'])

eval_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)
eval_docs = [[label.document for label in multi_label.labels] for multi_label in eval_labels]

# English and Greek SAS models 
advanced_eval_result = pipe.eval(labels=eval_labels, documents=eval_docs, sas_model_name_or_path="cross-encoder/stsb-roberta-large")
# advanced_eval_result = pipe.eval(labels=eval_labels, documents=eval_docs, sas_model_name_or_path="lighteternal/stsb-xlm-r-greek-transfer")

metrics = advanced_eval_result.calculate_metrics()
pipe.print_eval_report(advanced_eval_result)
print(metrics["Reader"]["sas"])