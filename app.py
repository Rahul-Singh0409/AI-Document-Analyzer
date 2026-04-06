import streamlit  as st 
import os 
import tempfile 

from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from rich import print 

load_dotenv()

# Creating the base configuration of the website
st.set_page_config(page_title="Knowledge Assistant" ,page_icon=":lobster:")
st.title(":books: Document Analyzer ...")
st.write("An Intelligent analyzer which help you understand your query")

# Creating a session with the streamlit \
if "vector_store"  not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages=[]

# creation of the side bar for uploading the Document and processing the Document 
with st.sidebar:
    st.header("Upload your favourite Document")
    Uploaded_file = st.file_uploader("Upload your Document" , type =".pdf")
    if st.button("Indexing the Document"):
        if Uploaded_file:
            with st.spinner("Analyzing the Document "):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(Uploaded_file.getvalue())
                    path = temp.name  

                    # Path has the document in the local storage 
                # Loading the document via PyPDF Loader 
                loader = PyPDFLoader(path) 
                document = loader.load()
                # Creating the chunks of this document 
                Splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splitted_chunks = Splitter.split_documents(document)

                # Need to create the vector Store for the session 
                # Creation of embeddings from the Open AI 
                embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
                st.session_state.vector_store= FAISS.from_documents(splitted_chunks, embedding=embeddings )
                # Confirming the User that the PDF has been uploaded and embeddings has been created 
                # This works (Windows: Win + . | Mac: Cmd + Ctrl + Space)
                st.success("PDF has been Indexed Successfully", icon="📚")
                os.remove(path)
        else:
                st.error("Please , Upload a file first for the documents")

######## Chat Display{Crucial Layer}########
for message in st.session_state.messages: # for the message in the above session ,we are travrsing in such a way that onemessage is from user and other from LLMs
    with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User Window ####
if prompt := st.chat_input("Ask any Question about the Document"): # This is a User Prompt 
                # User Input 
                with st.chat_message("user"):
                      st.markdown(prompt)
                st.session_state.messages.append({"role":"user","content":prompt})
                # LLM Output 
                # 1 . Test case ---> Whether the Input is being presented by the user o not 
                if st.session_state.vector_store is None :
                      st.warning("I don't have the Document , please Upload it from The SideMenu")

                else:
                      with st.chat_message("Assistant"):
                            with st.spinner("Searching the Document, Please grab a :coffee:"):
                                  # Now we have to create a retriever to get the desired docs from the document 
                                  retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k":4,"lambda_mult":0.5})
                                  # we will build a langchain 
                                  model = ChatOpenAI(model='gpt-5-nano')
                                  LLM_prompt = PromptTemplate.from_template(
                                        template = """You are an AI assistant which answers from the given {context} in 5 bullet points,
                                        Context:{context}
                                        Question:{question}
                                        """
                
                                  )
                                  def formatting_docs(document):
                                        context_text = "\n\n".join(doc.page_content for doc in document)
                                        return context_text
                                  Parser = StrOutputParser()

                                  # creating the chain 
                                  chain = ({'context':retriever| RunnableLambda(formatting_docs), 'question' : RunnablePassthrough()}|LLM_prompt|model|Parser)
                                  response = chain.invoke (prompt)
                                  st.markdown(response)
                                  st.session_state.messages.append({"role":"assistant", "content":response}) 

# The last line is appending the messages from the LLMs to the messages history in the particular Session 

