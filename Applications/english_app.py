%%writefile streamlit_app.py 
import streamlit as st 

st.title('Question & Knowledge Graph Generation Application')

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

def skipComputation():

  logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
  logging.getLogger("haystack").setLevel(logging.INFO)

  es_server = Popen(
    ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon
  )   

  print("ES starting now")
  st.write("Please wait 40 seconds for the ElasticSearch initialization.")
  time.sleep(40)
  print("ES has started")
  st.write("ElasticSearch initialization completed.")

  global question_generator
  global reader

  question_generator = QuestionGenerator(model_name_or_path = "valhalla/t5-base-e2e-qg", split_length = 256, split_overlap = 64, num_queries_per_doc = 1, use_gpu = True)
  reader = FARMReader(model_name_or_path="ahotrod/albert_xxlargev1_squad2_512", top_k = 1, max_seq_len = 384, doc_stride = 128, context_window_size = 256, confidence_threshold = 0.65)    

  # faster model
  # reader = FARMReader(model_name_or_path="distilbert-base-cased-distilled-squad", top_k = 1, max_seq_len = 384, doc_stride = 128, context_window_size = 256, confidence_threshold = 0.65)    

  nltk.download('punkt')

  return None

def kg_models():

  global model
  global reader_kg

  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  reader_kg = FARMReader(model_name_or_path="ahotrod/albert_xxlargev1_squad2_512", top_k = 1, max_seq_len = 384, doc_stride = 128, confidence_threshold = 0.4)

  return None

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

if "knowledge_graph" not in st.session_state:
    st.session_state["knowledge_graph"] = False

if "knowledge_graph_2" not in st.session_state:
    st.session_state["knowledge_graph_2"] = False

if "knowledge_graph_3" not in st.session_state:
    st.session_state["knowledge_graph_3"] = False

if "knowledge_graph_4" not in st.session_state:
    st.session_state["knowledge_graph_4"] = False 

st.write("What would you like to do?")

if st.button("Create a QA Dataset from Text."):

  st.session_state["qa_dataset"] = not st.session_state["qa_dataset"]

if st.button("Get Questions & Answers from Recent News."):

  st.session_state["news_qa"] = not st.session_state["news_qa"]

if st.button("Create a Knowledge Graph from Questions & Answers."):
  
  st.session_state["knowledge_graph"] = not st.session_state["knowledge_graph"] 

################################################################################ FIRST CHOICE

if st.session_state["qa_dataset"]:

  user_text = st.text_area("Input text for the creation of a QA Dataset:")

  if st.button("CREATE"):

    st.session_state["qa_dataset_2"] = not st.session_state["qa_dataset_2"]

    skipComputation()

    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()

    docs = [{"content": user_text}]
    document_store.write_documents(docs)
    
    qg_qa_result = []

    qg_result = []
    qa_result = []

    question_generation_pipeline = QuestionGenerationPipeline(question_generator)
    for idx, document in enumerate(tqdm(document_store)):

      print(f"\n * Generating questions for document {idx}: {document.content[:100]}...\n")
      qg_result.append(question_generation_pipeline.run(documents=[document]))
      print_questions(qg_result[idx])

      qg_number = 0
      
      for i in range(len(qg_result[idx]['generated_questions'][0]['questions'])):
        qa_result_2 = reader.predict(
            query=str(qg_result[idx]['generated_questions'][0]['questions'][i]),
            documents=[document],
            top_k=1)
        qa_result.append(qa_result_2)

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
          qa_number = qa_number + len(qg_result[j-1]['generated_questions'][0]['questions'])
      
      for i in range(len(qg_result[j]['generated_questions'][0]['questions'])):
        
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

      data.append({"title":"QA Dataset from User Input", "context":con, "question":ques, "answer":ans, "answer_start":ans_start})

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
        label="Download as JSON",
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

if 'new_knowledge_graph' not in st.session_state:
    st.session_state.new_knowledge_graph = None

if 'new_similarity' not in st.session_state:
    st.session_state.new_similarity = None  

if st.session_state["news_qa"]:

  user_text = st.text_input("Input entity or phrase for news search:")

  if st.button("SEARCH"):

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
          "lr": "lang_en",             
          "num": 8,        
          "tbs" : "qdr:y" 
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


    ########################
    # COREFERENCE RESOLUTION
    ########################

    import locale
    def getpreferredencoding(do_setlocale = True):
        return "UTF-8"
    locale.getpreferredencoding = getpreferredencoding

    # Define lightweight function for resolving references in text
    def resolve_references(doc: Doc) -> str:
        """Function for resolving references with the coref ouput
        doc (Doc): The Doc object processed by the coref pipeline
        RETURNS (str): The Doc string with resolved references
        """
        token_mention_mapper = {}
        output_string = ""
        clusters = [
            val for key, val in doc.spans.items() if key.startswith("coref_cluster")
        ]

        # Iterate through every found cluster
        for cluster in clusters:
            first_mention = cluster[0]
            # Iterate through every other span in the cluster
            for mention_span in list(cluster)[1:]:
                # Set first_mention as value for the first token in mention_span in the token_mention_mapper
                token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
                
                for token in mention_span[1:]:
                    # Set empty string for all the other tokens in mention_span
                    token_mention_mapper[token.idx] = ""

        # Iterate through every token in the Doc
        for token in doc:
            # Check if token exists in token_mention_mapper
            if token.idx in token_mention_mapper:
                output_string += token_mention_mapper[token.idx]
            # Else add original token text
            else:
                output_string += token.text + token.whitespace_

        return output_string
      
    ############################################################################

    coref = spacy.load("en_coreference_web_trf")

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
              doc = coref(single_url.text)
              new_text = resolve_references(doc)
              info.append(new_text)
              title.append(single_url.title)
        except:
            continue  
        
    st.session_state.title = title

    ############################################################################

    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()

    for i in range(len(info)):
      docs = [{"content": info[i]}]
      document_store.write_documents(docs)

    qg_qa_result = []
    qg_result = []
    qa_result = []

    question_generation_pipeline = QuestionGenerationPipeline(question_generator)
   
    for idx, document in enumerate(tqdm(document_store)):

      print(f"\n * Generating questions for document {idx}: {document.content[:100]}...\n")
      qg_result.append(question_generation_pipeline.run(documents=[document]))
      print_questions(qg_result[idx])
      
      for i in range(len(qg_result[idx]['generated_questions'][0]['questions'])):
        qa_result_2 = reader.predict(
            query=str(qg_result[idx]['generated_questions'][0]['questions'][i]),
            documents=[document],
            top_k=1)
        qa_result.append(qa_result_2)
        print_answers(qa_result_2)

    st.session_state.qg_result = qg_result
    st.session_state.qa_result = qa_result

    ############################################################################

    qa_strings = []
    qa_number = 0
    sources = []

    for j in range(len(qg_result)):
      
      from_link = str(title[j] + ', ' + links[j])
      
      if j != 0:
        qa_number = qa_number + len(qg_result[j-1]['generated_questions'][0]['questions'])

      st.write("**The generated questions and their respective answers from article no. ", str(j+1),  " are:**")
      st.write("Source:", from_link)

      for i in range(len(qg_result[j]['generated_questions'][0]['questions'])):
          r1 = str(qa_result[qa_number + i]['answers'])
          r = re.search("Answer {'answer': '(.*)', 'type':", r1)
      
          if r is not None:
            a = (str(qa_result[qa_number + i]['query']))
            b = a + ' ' + str(r.group(1)) + '.'
            qa_strings.append(b)
            sources.append(from_link)

            st.write('Question:', a)
            st.write('Answer:', str(r.group(1)))

      st.session_state.qa_strings = qa_strings

###############################################################################

if st.session_state["news_qa"] and st.session_state["news_qa_2"]:

    if st.button("Create a new QA Dataset from the above."):

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
              qa_number = qa_number + len(qg_result[j-1]['generated_questions'][0]['questions'])
            
          for i in range(len(qg_result[j]['generated_questions'][0]['questions'])):
              
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
            label="Download as JSON",
            file_name="dataset.json",
            mime="application/json_dataset",
            data=json_string,
        )

################################################################################ THIRD CHOICE

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
import wikipedia
import IPython
from pyvis.network import Network

from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

if st.session_state["knowledge_graph"]:

  user_text = st.text_input("Input entity or phrase for news search or keep previously searched phrase:")

  if st.button("NEW SEARCH (base info e.g. wikipedia included)"):

    st.session_state["knowledge_graph_2"] = not st.session_state["knowledge_graph_2"]

    skipComputation()

    ###############
    # NEWS SCRAPING
    ###############

    ns = []
    links = [] 

    params = {
      "api_key": "34babb77f331d1ece1d050512f525a4c8d65dde9045a4d51c88eae0b4aa155b0",  # your serpapi api key
      "engine": "google",         # serpapi parsing engine
      "q": user_text,             # search query
      "lr": "en",             
      "num": 3       
    }
    
    search = GoogleSearch(params)   # where data extraction happens on the backend
    results = search.get_dict()     # JSON - > Python dictionary

    for result in results['organic_results']:
      links.append(result['link'])
      print(result['link'])

    # without NER
    params = {
          "api_key": "34babb77f331d1ece1d050512f525a4c8d65dde9045a4d51c88eae0b4aa155b0",  # your serpapi api key
          "engine": "google", 
          "tbm": "nws",               # serpapi parsing engine
          "q": user_text,             # search query
          "lr": "lang_en",             
          "num": 8,        
          "tbs" : "qdr:y" 
        }
    
    search = GoogleSearch(params)   # where data extraction happens on the backend
    results = search.get_dict()     # JSON - > Python dictionary
    ns.append(results["news_results"])

    # without NER
    for i in range(len(ns[0])):
      print(ns[0][i]["link"])
      links.append(ns[0][i]["link"])

    ########################
    # COREFERENCE RESOLUTION
    ########################

    import locale
    def getpreferredencoding(do_setlocale = True):
        return "UTF-8"
    locale.getpreferredencoding = getpreferredencoding

    # Define lightweight function for resolving references in text
    def resolve_references(doc: Doc) -> str:
        """Function for resolving references with the coref ouput
        doc (Doc): The Doc object processed by the coref pipeline
        RETURNS (str): The Doc string with resolved references
        """
        token_mention_mapper = {}
        output_string = ""
        clusters = [
            val for key, val in doc.spans.items() if key.startswith("coref_cluster")
        ]

        # Iterate through every found cluster
        for cluster in clusters:
            first_mention = cluster[0]
            # Iterate through every other span in the cluster
            for mention_span in list(cluster)[1:]:
                # Set first_mention as value for the first token in mention_span in the token_mention_mapper
                token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
                
                for token in mention_span[1:]:
                    # Set empty string for all the other tokens in mention_span
                    token_mention_mapper[token.idx] = ""

        # Iterate through every token in the Doc
        for token in doc:
            # Check if token exists in token_mention_mapper
            if token.idx in token_mention_mapper:
                output_string += token_mention_mapper[token.idx]
            # Else add original token text
            else:
                output_string += token.text + token.whitespace_

        return output_string
      
    ############################################################################

    coref = spacy.load("en_coreference_web_trf")

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
              doc = coref(single_url.text)
              new_text = resolve_references(doc)
              info.append(new_text)
              title.append(single_url.title)
        except:
            continue  
        
    st.session_state.title = title

    ############################################################################

    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()

    for i in range(len(info)):
      docs = [{"content": info[i]}]
      document_store.write_documents(docs)

    qg_qa_result = []
    qg_result = []
    qa_result = []

    question_generation_pipeline = QuestionGenerationPipeline(question_generator)
   
    for idx, document in enumerate(tqdm(document_store)):

      print(f"\n * Generating questions for document {idx}: {document.content[:100]}...\n")
      qg_result.append(question_generation_pipeline.run(documents=[document]))
      print_questions(qg_result[idx])

      qg_number = 0
      
      for i in range(len(qg_result[idx]['generated_questions'][0]['questions'])):
        qa_result_2 = reader.predict(
            query=str(qg_result[idx]['generated_questions'][0]['questions'][i]),
            documents=[document],
            top_k=1)
        qa_result.append(qa_result_2)
        print_answers(qa_result_2)

    ############################################################################

    qa_strings = []
    qa_number = 0
    sources = []

    for j in range(len(qg_result)):
      
      from_link = str(title[j] + ', ' + links[j])
      
      if j != 0:
        qa_number = qa_number + len(qg_result[j-1]['generated_questions'][0]['questions'])
      
      st.write("**The generated questions and their respective answers from article no. ", str(j),  " are:**")
      st.write("Source:", from_link)

      for i in range(len(qg_result[j]['generated_questions'][0]['questions'])):
          r1 = str(qa_result[qa_number + i]['answers'])
          r = re.search("Answer {'answer': '(.*)', 'type':", r1)
      
          if r is not None:
            a = (str(qa_result[qa_number + i]['query']))
            b = a + ' ' + str(r.group(1)) + '.'
            qa_strings.append(b)
            sources.append(from_link)

            st.write('Question:', a)
            st.write('Answer:', str(r.group(1)))

    ########################
    # REBEL KNOWLEDGE GRAPH
    ########################

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    for_similarity = []
    new_qa_strings = []

    def extract_relations_from_model_output(text):
        relations = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")

        for token in text_replaced.split():

            if token == "<triplet>":
                current = 't'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                    relation = ''
                subject = ''

            elif token == "<subj>":
                current = 's'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                object_ = ''

            elif token == "<obj>":
                current = 'o'
                relation = ''

            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token

        if subject != '' and relation != '' and object_ != '':
            relations.append({
                'head': subject.strip(),
                'type': relation.strip(),
                'tail': object_.strip()
            })

            for_similarity.append(relations[0]['head'] + ' ' + relations[0]['type'] + ' ' + relations[0]['tail'])
            print("String for similarity created")

        return relations

    class KB():
        def __init__(self):
            self.relations = []

        def are_relations_equal(self, r1, r2):
            return (r1[0] == r2[0])

        def exists_relation(self, r1):
            return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

        def add_relation(self, r, i):
            if not self.exists_relation(r):
              self.relations.append(r)
              new_qa_strings.append(qa_strings[i])
            else:
              for_similarity.pop()

        def print(self):
            print("Relations:")
            for r in self.relations:
                print(f"  {r}")
            
    def from_small_text_to_kb(text, verbose=False):
        kb = KB()

        for i in range(len(text)):
            
            print("\n") 
            print(text[i])

            # Tokenizer text
            model_inputs = tokenizer(text[i], max_length=512, padding=True, truncation=True,
                                return_tensors='pt')
            if verbose:
                print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

            # Generate
            gen_kwargs = {
                "max_length": 256,
                "length_penalty": 0,
                "num_beams": 3,
                "num_return_sequences": 1
            }
            generated_tokens = model.generate(
                **model_inputs,
                **gen_kwargs,
            )
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            print(decoded_preds)

            # create kb
            for sentence_pred in decoded_preds:
                relations = extract_relations_from_model_output(sentence_pred)
                if relations:
                    kb.add_relation(relations, i)

        return kb

    knowledge_graph = [] 

    kb = from_small_text_to_kb(qa_strings, verbose=True)
    for i in range(len(kb.relations)):
      knowledge_graph.append(kb.relations[i][0])

    kb.print()

    ############################################
    # SIMILARITY SCORES BETWEEN TRIPLETS AND QAs
    ############################################

    from sentence_transformers import SentenceTransformer, util
    import torch.nn.functional as F

    new_knowledge_graph = []
    new_similarity = []

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for i in range(len(new_qa_strings)):
        embedding1 = model.encode(new_qa_strings[i])
        embedding2 = model.encode(for_similarity[i])
        embeddings = util.pytorch_cos_sim(embedding1, embedding2)
        print(new_qa_strings[i])
        print(for_similarity[i])
        print(str(embeddings) + '\n')
        if embeddings > 0.65:
          new_knowledge_graph.append(knowledge_graph[i])
          new_similarity.append(for_similarity[i])

    st.header("Relations:")
    st.write(new_knowledge_graph)

    st.session_state.new_knowledge_graph = new_knowledge_graph
    st.session_state.new_similarity = new_similarity

    def save_network_html(new_knowledge_graph, filename="network.html"):
      # create network
      net = Network(directed=True, width="1920px", height="1080px", bgcolor="#eeeeee", notebook=True)

      # nodes
      color_entity = "#F1DDCF"

      # edges
      for r in new_knowledge_graph:
          net.add_node(r["head"], shape="circle", color=color_entity)
          net.add_node(r["tail"], shape="circle", color=color_entity)
          net.add_edge(r["head"], r["tail"],
                      title=r["type"], label=r["type"])
          
      # save network
      net.repulsion(
          node_distance=200,
          central_gravity=0.2,
          spring_length=200,
          spring_strength=0.05,
          damping=0.09
      )
      net.set_edge_smooth('dynamic')
      net.show(filename)

    filename = "my_kg.html"
    save_network_html(new_knowledge_graph, filename=filename)

    import streamlit.components.v1 as components

    st.header("KNOWLEDGE GRAPH VISUALIZED")

    HtmlFile = open("my_kg.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 1080)

  if st.button("PREVIOUSLY SEARCHED PHRASE (base info e.g. wikipedia excluded)"):

    st.session_state["knowledge_graph_3"] = not st.session_state["knowledge_graph_3"]

    qa_strings = st.session_state.qa_strings

    ########################
    # REBEL KNOWLEDGE GRAPH
    ########################

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    for_similarity = []
    new_qa_strings = []

    def extract_relations_from_model_output(text):
        relations = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")

        for token in text_replaced.split():

            if token == "<triplet>":
                current = 't'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                    relation = ''
                subject = ''

            elif token == "<subj>":
                current = 's'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                object_ = ''

            elif token == "<obj>":
                current = 'o'
                relation = ''

            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token

        if subject != '' and relation != '' and object_ != '':
            relations.append({
                'head': subject.strip(),
                'type': relation.strip(),
                'tail': object_.strip()
            })

            for_similarity.append(relations[0]['head'] + ' ' + relations[0]['type'] + ' ' + relations[0]['tail'])
            print("String for similarity created")

        return relations

    class KB():
        def __init__(self):
            self.relations = []

        def are_relations_equal(self, r1, r2):
            return (r1[0] == r2[0])

        def exists_relation(self, r1):
            return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

        def add_relation(self, r, i):
            if not self.exists_relation(r):
              self.relations.append(r)
              new_qa_strings.append(qa_strings[i])
            else:
              for_similarity.pop()

        def print(self):
            print("Relations:")
            for r in self.relations:
                print(f"  {r}")
            
    def from_small_text_to_kb(text, verbose=False):
        kb = KB()

        for i in range(len(text)):
            
            print("\n") 
            print(text[i])

            # Tokenizer text
            model_inputs = tokenizer(text[i], max_length=512, padding=True, truncation=True,
                                return_tensors='pt')
            if verbose:
                print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

            # Generate
            gen_kwargs = {
                "max_length": 256,
                "length_penalty": 0,
                "num_beams": 3,
                "num_return_sequences": 1
            }
            generated_tokens = model.generate(
                **model_inputs,
                **gen_kwargs,
            )
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            print(decoded_preds)

            # create kb
            for sentence_pred in decoded_preds:
                relations = extract_relations_from_model_output(sentence_pred)
                if relations:
                    kb.add_relation(relations, i)

        return kb

    knowledge_graph = [] 

    kb = from_small_text_to_kb(qa_strings, verbose=True)
    for i in range(len(kb.relations)):
      knowledge_graph.append(kb.relations[i][0])

    kb.print()

    ############################################
    # SIMILARITY SCORES BETWEEN TRIPLETS AND QAs
    ############################################

    from sentence_transformers import SentenceTransformer, util
    import torch.nn.functional as F

    new_knowledge_graph = []
    new_similarity = []

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for i in range(len(new_qa_strings)):
        embedding1 = model.encode(new_qa_strings[i])
        embedding2 = model.encode(for_similarity[i])
        embeddings = util.pytorch_cos_sim(embedding1, embedding2)
        print(new_qa_strings[i])
        print(for_similarity[i])
        print(str(embeddings) + '\n')
        if embeddings > 0.65:
          new_knowledge_graph.append(knowledge_graph[i])
          new_similarity.append(for_similarity[i])

    st.header("Relations:")
    st.write(new_knowledge_graph)

    st.session_state.new_knowledge_graph = new_knowledge_graph
    st.session_state.new_similarity = new_similarity

    def save_network_html(new_knowledge_graph, filename="network.html"):
      # create network
      net = Network(directed=True, width="1000px", height="1000px", bgcolor="#eeeeee", notebook=True)

      # nodes
      color_entity = "#F1DDCF"

      # edges
      for r in new_knowledge_graph:
          net.add_node(r["head"], shape="circle", color=color_entity)
          net.add_node(r["tail"], shape="circle", color=color_entity)
          net.add_edge(r["head"], r["tail"],
                      title=r["type"], label=r["type"])
          
      # save network
      net.repulsion(
          node_distance=200,
          central_gravity=0.2,
          spring_length=200,
          spring_strength=0.05,
          damping=0.09
      )
      net.set_edge_smooth('dynamic')
      net.show(filename)

    filename = "my_kg.html"
    save_network_html(new_knowledge_graph, filename=filename)

    import streamlit.components.v1 as components

    st.header("VISUALIZED KNOWLEDGE GRAPH")

    HtmlFile = open("my_kg.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 1080)

if st.session_state["knowledge_graph"] and st.session_state["knowledge_graph_2"]:

    user_text = st.text_input("Input question for the Knowledge Graph:")

    if st.button("ASK"):

        st.session_state["knowledge_graph_4"] = not st.session_state["knowledge_graph_4"]

        kg_models()

        new_knowledge_graph = st.session_state.new_knowledge_graph
        new_similarity = st.session_state.new_similarity

        document_store = ElasticsearchDocumentStore()
        document_store.delete_documents()

        question = user_text

        for i in range(len(new_knowledge_graph)):
            embedding1 = model.encode(question)
            embedding2 = model.encode(new_similarity[i])
            embeddings = util.pytorch_cos_sim(embedding1, embedding2)
            if embeddings.item() > 0.2:
                print(new_similarity[i])
                print(embeddings.item())
                docs = [{"content": new_similarity[i]}]     
                document_store.write_documents(docs)

        answers = []

        for idx, document in enumerate(tqdm(document_store)):
            answers.append(reader_kg.predict(
                query=question,
                documents=[document],
                top_k=1
            ))

        import pandas as pd

        possible_answers = []
        scores = []

        for i in range(len(answers)):

          print(answers[i])
          answer_temp = str(answers[i]['answers'])
          answer = re.search("Answer {'answer': '(.*)', 'type':", answer_temp)
          score = re.search(", 'score': (.*), 'context':", answer_temp)

          if answer is not None:
              possible_answers.append(answer.group(1))
              scores.append(float(score.group(1)))

        answer_list = pd.DataFrame(
        {'possible_answers': possible_answers,
         'scores': scores}
        )

        answer_list.sort_values(by=['scores'], inplace=True, ascending=False)

        st.write("Possible Answers:")
        print(answer_list)
        st.dataframe(answer_list)

if st.session_state["knowledge_graph"] and st.session_state["knowledge_graph_3"]:

    user_text = st.text_input("Input question for the Knowledge Graph:")

    if st.button("ASK"):

        st.session_state["knowledge_graph_4"] = not st.session_state["knowledge_graph_4"]

        kg_models()

        new_knowledge_graph = st.session_state.new_knowledge_graph
        new_similarity = st.session_state.new_similarity    

        document_store = ElasticsearchDocumentStore()
        document_store.delete_documents()

        question = user_text

        for i in range(len(new_knowledge_graph)):
            embedding1 = model.encode(question)
            embedding2 = model.encode(new_similarity[i])
            embeddings = util.pytorch_cos_sim(embedding1, embedding2)
            if embeddings.item() > 0.2:
                print(new_similarity[i])
                print(embeddings.item())
                docs = [{"content": new_similarity[i]}]     
                document_store.write_documents(docs)

        answers = []

        for idx, document in enumerate(tqdm(document_store)):
            answers.append(reader_kg.predict(
                query=question,
                documents=[document],
                top_k=1
            ))

        import pandas as pd

        possible_answers = []
        scores = []

        for i in range(len(answers)):

          print(answers[i])
          answer_temp = str(answers[i]['answers'])
          answer = re.search("Answer {'answer': '(.*)', 'type':", answer_temp)
          score = re.search(", 'score': (.*), 'context':", answer_temp)

          if answer is not None:
              possible_answers.append(answer.group(1))
              scores.append(float(score.group(1)))

        answer_list = pd.DataFrame(
        {'possible_answers': possible_answers,
         'scores': scores}
        )

        answer_list.sort_values(by=['scores'], inplace=True, ascending=False)

        st.write("Possible Answers:")
        print(answer_list)
        st.dataframe(answer_list)