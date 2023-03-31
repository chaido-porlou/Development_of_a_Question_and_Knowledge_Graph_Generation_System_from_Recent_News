%%writefile streamlit_app_greek.py 
import streamlit as st 

st.title('Εφαρμογή Παραγωγής Ερωτήσεων & Απαντήσεων')

import time
import os
from subprocess import Popen, PIPE, STDOUT

# Imports needed to run this notebook

from pprint import pprint
from tqdm import tqdm
from haystack.nodes import QuestionGenerator, FARMReader
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import QuestionGenerationPipeline
from haystack.utils import launch_es, print_questions, print_answers

import logging

import nltk 

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5TokenizerFast
import textwrap

def skipComputation():

  logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
  logging.getLogger("haystack").setLevel(logging.INFO)

  es_server = Popen(
    ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon
  )   

  print("ES starting now")
  st.write("Παρακαλώ περιμένετε 40 δευτερόλεπτα για σύνδεση με την πλατφόρμα ElasticSearch.")
  time.sleep(40)
  print("ES has started")
  st.write("Η σύνδεση με την πλατφόρμα ElasticSearch ολοκληρώθηκε.")

  global question_generator
  global reader
  global tokenizer
  global hfmodel

  hfmodel = AutoModelForSeq2SeqLM.from_pretrained("chaido13/greek-m2m100-4ep-384", use_auth_token=True)
  print("ready model")
  tokenizer = AutoTokenizer.from_pretrained("chaido13/greek-m2m100-4ep-384")
  print("ready tokenizer")
  reader = FARMReader(model_name_or_path="Danastos/squad_bert_el", top_k = 1, max_seq_len = 384, doc_stride = 128, context_window_size = 256, confidence_threshold = 0.3, use_auth_token = "")    
  print("ready reader")

  nltk.download('punkt')

  return None

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

######################################################## STARTING SESSION STATES

if "qa_dataset" not in st.session_state:
    st.session_state["qa_dataset"] = False

if "qa_dataset_2" not in st.session_state:
    st.session_state["qa_dataset_2"] = False

if "news_qa" not in st.session_state:
    st.session_state["news_qa"] = False

if "news_qa_2" not in st.session_state:
    st.session_state["news_qa_2"] = False

if "news_qa_3" not in st.session_state:
    st.session_state["news_qa_3"] = False  

st.write("Τι θα θέλατε να κάνετε;")

if st.button("Δημιουργία Συνόλου Δεδομένων Ερώτησης - Απάντησης από Κείμενο."):

  st.session_state["qa_dataset"] = not st.session_state["qa_dataset"]

if st.button("Παραγωγή Ερωτήσεων & Απαντήσεων από Πρόσφατα Νέα."):

  st.session_state["news_qa"] = not st.session_state["news_qa"]

st.session_state["run_id"] = 0

################################################################################ FIRST CHOICE

if st.session_state["qa_dataset"]:

  user_text = st.text_area("Εισάγετε κείμενο για τη δημιουργία του Συνόλου Δεδομένων:")

  if st.button("ΔΗΜΙΟΥΡΓΙΑ"):

    st.session_state["qa_dataset_2"] = not st.session_state["qa_dataset_2"]

    skipComputation()

    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()

    docs_for_qg = []

    docs = [{"content": user_text}]
    document_store.write_documents(docs)
    docs_for_qg.append(user_text)
    
    qg_result = []
    qa_result = []

    for idx, document in enumerate(tqdm(document_store)):

      questions = []

      print(f"\n * Generating questions for document {idx}: {document.content[:100]}...\n")
      lines = textwrap.wrap(docs_for_qg[idx], 384, break_long_words=False, fix_sentence_endings=True)

      for j in range(len(lines)):
        questions.append(hf_run_model(lines[j]))

      print(questions)

      temp_list = []

      for m in range(len(questions)):
        for k in range(len(questions[m])):
          temp = questions[m][k][0].split('; ')
          for p in range(len(temp)-1):
            temp_list.append(temp[p] + ';')
      
      qg_result.append(temp_list)

      print(qg_result[idx])

      for i in range(len(qg_result[idx])):
        qa_result_2 = reader.predict(
            query=str(qg_result[idx][i]),
            documents=[document],
            top_k=1)
        qa_result.append(qa_result_2)
        print(qa_result_2)

    #####################
    # QA DATASET CREATION
    #####################

    import pandas as pd
    import json
    import uuid
    import re

    data = []
    qa_number = 0

    for j in range(len(qg_result)):

      ans = []
      con =[]
      ans_start = []
      ques = []

      if j != 0:
          qa_number = qa_number + len(qg_result[j-1])
      
      for i in range(len(qg_result[j])):
        
        print(str(qa_result[qa_number + i]['answers']))
        answer1 = str(qa_result[qa_number + i]['answers'])
        answer = re.search("Answer {'answer': '(.*)', 'type':", answer1)

        if answer is not None:
          ans.append(answer.group(1))

          context1 = str(qa_result[qa_number + i]['answers'])
          context = re.search(", 'context': (.*), 'offsets_in_document':", context1)
          context = (str(context.group(1)))[1:-1]
          con.append(context)

          answer_start1 = str(qa_result[qa_number + i]['answers'])
          answer_start = re.search(", 'offsets_in_context': \[\{'start': (.*), 'end':", context1)
          ans_start.append(int(answer_start.group(1)))

          ques.append(str(qa_result[qa_number + i]['query']))

      data.append({"title":"Σύνολο Δεδομένων Ερώτησης - Απάντησης από Δοθέν Κείμενο", "context":con, "question":ques, "answer":ans, "answer_start":ans_start})

    data1 = pd.DataFrame(data)

    # assume data is your dataframe
    data = {"data": []}

    for index, row in data1.iterrows():
        article = {"title": row["title"], "paragraphs": []}

        for i in range(len(data1.loc[index, 'context'])):
          context = {"qas": [], "context": row["context"][i]}
          question = {"question": row["question"][i], "id": str(uuid.uuid4()), "answers": [{"text": row["answer"][i], "answer_start": row["answer_start"][i]}]}
          context["qas"].append(question)
          article["paragraphs"].append(context)
        
        data["data"].append(article)

    with open('dataset.json', 'w') as f:
        json.dump(data, f)

    json_string = json.dumps(data)

    st.json(json_string, expanded=True)

    st.download_button(
        label="Λήψη σε μορφή JSON.",
        file_name="dataset.json",
        mime="application/json_dataset",
        data=json_string,
    )

################################################################################ SECOND CHOICE

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from serpapi import GoogleSearch
import json
import spacy
from spacy import displacy

import itertools

from spacy.tokens import Doc
import pandas as pd

import spacy_transformers

from newspaper import Article
from newspaper import Config

import re

if 'qg_result' not in st.session_state:
    st.session_state.qg_result = None

if 'qa_result' not in st.session_state:
    st.session_state.qa_result = None   

if 'title' not in st.session_state:
    st.session_state.title = None

if 'qa_strings' not in st.session_state:
    st.session_state.qa_strings = None

if st.session_state["news_qa"]:

  user_text = st.text_input("Εισάγετε Φράση για Αναζήτηση Ειδήσεων:")

  if st.button("ΑΝΑΖΗΤΗΣΗ"):

    st.session_state["news_qa_2"] = not st.session_state["qa_dataset_2"]

    skipComputation()

    ###############
    # NEWS SCRAPING
    ###############

    # without NER
    params = {
          "api_key": "34babb77f331d1ece1d050512f525a4c8d65dde9045a4d51c88eae0b4aa155b0",  # your serpapi api key
          "engine": "google", 
          "tbm": "nws",               # serpapi parsing engine
          "q": user_text,             # search query
          "lr": "el",             
          "num": 8,        
          "gl" : "gr" 
        }
        
    ns = []
    
    search = GoogleSearch(params)   # where data extraction happens on the backend
    results = search.get_dict()     # JSON - > Python dictionary
    ns.append(results["news_results"])

    links = []  

    # without NER
    for i in range(len(ns[0])):
      print(ns[0][i]["link"])
      links.append(ns[0][i]["link"])
      
    ############################################################################

    info = []
    title = []
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    config = Config()
    config.browser_user_agent = user_agent

    for i in range(len(links)):
        single_url = Article(links[i],config=config)
        try:
            single_url.download()
            single_url.parse()
            single_url.nlp()
            if "Virgin Islands" not in single_url.text:
              info.append(single_url.text)
              title.append(single_url.title)
        except:
            continue  
        
    st.session_state.title = title

    ############################################################################

    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()

    docs_for_qg = []

    for i in range(len(info)):
      docs = [{"content": info[i]}]
      document_store.write_documents(docs)
      docs_for_qg.append(info[i])
    
    qg_result = []
    qa_result = []

    for idx, document in enumerate(tqdm(document_store)):

      questions = []

      print(f"\n * Generating questions for document {idx}: {document.content[:100]}...\n")
      lines = textwrap.wrap(docs_for_qg[idx], 384, break_long_words=False, fix_sentence_endings=True)

      for j in range(len(lines)):
        questions.append(hf_run_model(lines[j]))

      temp_list = []

      for m in range(len(questions)):
        for k in range(len(questions[m])):
          temp = questions[m][k][0].split('; ')
          for p in range(len(temp)-1):
            temp_list.append(temp[p] + ';')
      
      qg_result.append(temp_list)

      print(qg_result[idx])

      for i in range(len(qg_result[idx])):
        qa_result_2 = reader.predict(
            query=str(qg_result[idx][i]),
            documents=[document],
            top_k=1)
        qa_result.append(qa_result_2)
        print(qa_result_2)

    st.session_state.qg_result = qg_result
    st.session_state.qa_result = qa_result

    ############################################################################

    qa_strings = []
    qa_number = 0
    sources = []

    for j in range(len(qg_result)):
      
      from_link = str(title[j] + ', ' + links[j])
      
      if j != 0:
        qa_number = qa_number + len(qg_result[j-1])

      st.write("**Οι παραχθείσες ερωτήσεις και οι αντίστοιχες απαντήσεις τους για το άρθρο με αριθμό ", str(j+1),  " είναι οι:**")
      st.write("Πηγή:", from_link)

      for i in range(len(qg_result[j])):
          r1 = str(qa_result[qa_number + i]['answers'])
          r = re.search("Answer {'answer': '(.*)', 'type':", r1)
      
          if r is not None:
            a = (str(qa_result[qa_number + i]['query']))
            b = a + ' ' + str(r.group(1)) + '.'
            qa_strings.append(b)
            sources.append(from_link)

            st.write('Ερώτηση:', a)
            st.write('Απάντηση:', str(r.group(1)))

      st.session_state.qa_strings = qa_strings

###############################################################################

if st.session_state["news_qa"] and st.session_state["news_qa_2"]:

    if st.button("Δημιουργία συνόλου δεδομένων Ερώτησης - Απάντησης από τα παραπάνω."):

        st.session_state["news_qa_3"] = not st.session_state["news_qa_3"]

        import pandas as pd
        import json
        import uuid
        import re

        data = []
        qa_number = 0

        qg_result = st.session_state.qg_result
        qa_result = st.session_state.qa_result
        title = st.session_state.title

        for j in range(len(qg_result)):

          title_new = title[j]

          ans = []
          con =[]
          ans_start = []
          ques = []

          if j != 0:
              qa_number = qa_number + len(qg_result[j-1])
      
          for i in range(len(qg_result[j])):
              
            answer1 = str(qa_result[qa_number + i]['answers'])
            answer = re.search("Answer {'answer': '(.*)', 'type':", answer1)

            if answer is not None:
              ans.append(answer.group(1))

              context1 = str(qa_result[qa_number + i]['answers'])
              context = re.search(", 'context': (.*), 'offsets_in_document':", context1)
              context = (str(context.group(1)))[1:-1]
              con.append(context)

              answer_start1 = str(qa_result[qa_number + i]['answers'])
              answer_start = re.search(", 'offsets_in_context': \[\{'start': (.*), 'end':", context1)
              ans_start.append(int(answer_start.group(1)))

              ques.append(str(qa_result[qa_number + i]['query']))

          data.append({"title":title_new, "context":con, "question":ques, "answer":ans, "answer_start":ans_start})

        data1 = pd.DataFrame(data)

        # assume data is your dataframe
        data = {"data": []}

        for index, row in data1.iterrows():
            article = {"title": row["title"], "paragraphs": []}

            for i in range(len(data1.loc[index, 'context'])):
              context = {"qas": [], "context": row["context"][i]}
              question = {"question": row["question"][i], "id": str(uuid.uuid4()), "answers": [{"text": row["answer"][i], "answer_start": row["answer_start"][i]}]}
              context["qas"].append(question)
              article["paragraphs"].append(context)
              
            data["data"].append(article)

        with open('dataset.json', 'w') as f:
            json.dump(data, f)

        json_string = json.dumps(data)

        st.json(json_string, expanded=True)

        st.download_button(
            label="Λήψη σε μορφή JSON.",
            file_name="dataset.json",
            mime="application/json_dataset",
            data=json_string,
        )
