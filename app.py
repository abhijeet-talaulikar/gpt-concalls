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
st.set_page_config(layout="wide")
st.title('Navigate Company Concall Transcripts')
openai_api_key = st.sidebar.text_input('OpenAI API Key', value='')
if "response_default" not in st.session_state:
  st.session_state['response_default'] = None
#endregion

#region Load PDF Function
def read_pdf(file, on_disk=True):
  if on_disk:
    loader = PyPDFLoader(file)
  else:
    loader = OnlinePDFLoader(file)
  documents = loader.load()
  return documents
#endregion

#region Prompts
prompt_default = '''
This document is the transcript of an earnings conference call of a company. Assume you are an analyst who 
attended this call. Identify which company this document is talking about. Create a report that answers each of the following 10 questions. 

What are some of the current and looming threats to the business? 
What is the debt level or debt ratio of the company right now? 
How do you feel about the upcoming product launches or new products? 
How are you managing or investing in your human capital? 
How do you track the trends in your industry? 
Are there major slowdowns in the production of goods? 
How will you maintain or surpass this performance in the next few quarters? 
What will your market look like in five years as a result of using your product or service? 
How are you going to address the risks that will affect the long-term growth of the company? 
How is the performance this quarter going to affect the long-term goals of the company? 


I want the output in markdown format. Here is an example of the format.

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
  uploaded_file = st.file_uploader("Choose a file")
  st.markdown('or')
  url_file = st.text_input('URL to file', '')
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
if frm_file_submitted and (uploaded_file is not None or url_file!=""):
  if not openai_api_key.startswith('sk-'):
      st.warning('Please enter your OpenAI API key!', icon='⚠')
  else:
    if uploaded_file is not None:
      documents = read_pdf(uploaded_file, True)
    elif url_file!="":
      documents = read_pdf(url_file, False)

    response_default = generate_default_response(documents)
    st.session_state['documents'] = pickle.dumps(documents)
    st.session_state['response_default'] = response_default

    updateChatBox()
    updateDefaultBox()
#endregion