!pip install streamlit
!ngrok authtoken 2MKnhNOl39k7Nto89zUoM3EGmPM_3CDGtNpaDJoJmuz1VUDy4
!pip install pyngrok 

from pyngrok import ngrok 
public_url = ngrok.connect(port='8501')
public_url

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

!pip install transformers[sentencepiece]

!pip install spacy-experimental
!pip install requests bs4 google-search-results
!pip install GoogleNews

!pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl

!pip install newspaper3k
!pip install spacy-transformers

!pip install farm-haystack

!wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
!tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
!chown -R daemon:daemon elasticsearch-7.9.2

! pip install wikipedia
! pip install pyvis
! pip install transformers

! pip install -U sentence-transformers

#########################################################################################################################################

!streamlit run /content/streamlit_app.py & npx localtunnel --port 8501