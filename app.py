__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#region Imports
import streamlit as st
import pickle
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
#endregion

#region Config
st.set_page_config(layout="wide", page_title="Concall Transcripts with GPT")
st.title('Concall Transcripts with GPT')
openai_api_key = st.sidebar.text_input('OpenAI API Key', value='')
if "response_default" not in st.session_state:
  st.session_state['response_default'] = None
#endregion

#region Load PDF Function
def read_pdf(file, on_disk=True):
  loader = OnlinePDFLoader(file)
  documents = loader.load()
  return documents
#endregion

#region Prompts
prompt_default = '''
This document is the transcript of an earnings conference call of a company. Assume you are an analyst who 
attended this call. Identify which company this document is talking about. Identify 10 best questions and their 
answers that would help summarize the company's performance. 
Create a report in a markdown format that answers each of those 10 questions. Here is an example of the format.

Example

## Insert company name here

**What are some of the current and looming threats to the business?** \\
Sample answer \\ \\

**What is the debt level or debt ratio of the company right now?** \\
Sample answer \\ \\
'''
#endregion

#region Query function
def get_guery_function(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
  texts = text_splitter.split_documents(documents)

  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
  vectordb = Chroma.from_documents(documents=texts, 
                                  embedding=embeddings,
                                  persist_directory=".")
  vectordb.persist()

  retriever = vectordb.as_retriever(search_kwargs={"k": 3})
  llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

  return qa
#endregion

#region Response Functions
def generate_custom_response(documents, question):
  qa = get_guery_function(documents)
  try:
      llm_response = qa(question)
      response = llm_response["result"]
  except Exception as err:
      response = str(err)
  return response

def generate_default_response(documents, prompt=prompt_default):
  qa = get_guery_function(documents)
  try:
      llm_response = qa(prompt)
      response = llm_response["result"]
  except Exception as err:
      response = str(err)
  return response
#endregion

#region UI
if "disabled" not in st.session_state:
    st.session_state.disabled = False

def disable():
  st.session_state.disabled = True

with st.form('frm_file'):
  url_file = st.text_input('URL to PDF file', '')
  frm_file_submitted = st.form_submit_button('Search', on_click=disable, disabled=st.session_state.disabled)
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')

op_custom = st.container()
op_default = st.container()

def updateDefaultBox():
  op_default.markdown(st.session_state['response_default'])

def updateChatBox():
  if st.session_state['response_default']:
    frm_ask = op_custom.form('frm_ask')
    question = frm_ask.text_area('If you have any specific question, ask here.', 'What are the three takeaways from this quarter?')
    frm_ask_submitted = frm_ask.form_submit_button('Ask')
    if frm_ask_submitted:
      if not openai_api_key.startswith('sk-'):
        op_custom.warning('Please enter your OpenAI API key!', icon='⚠')
      else:
        response_custom = generate_custom_response(
          pickle.loads(st.session_state['documents']), 
          question
          )  
        op_custom.info(response_custom)

if st.session_state['response_default']:
  updateChatBox()
  updateDefaultBox()

#endregion

#region Callback
if frm_file_submitted and url_file!="":
  if not openai_api_key.startswith('sk-'):
      st.warning('Please enter your OpenAI API key!', icon='⚠')
  else:
    documents = read_pdf(url_file, False)
    response_default = generate_default_response(documents)
    st.session_state['documents'] = pickle.dumps(documents)
    st.session_state['response_default'] = response_default

    updateChatBox()
    updateDefaultBox()
#endregion
