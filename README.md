# Document Analyzer
A high-performance Retrieval-Augmented Generation (RAG) application that allows users to upload complex documents (like SOPs) and perform intelligent, context-aware querying. Built with LangChain, OpenAI, and FAISS to demonstrate professional AI systems architecture.

It uses RAG (Retrieval- Augumentation and Generation Technique) to retrieved the docs , then we filter the docs and join them to give the context and using LLM API to generate the response . 

Some Features of the Above Analyzer -

Intelligent PDF Processing: Uses PyPDFLoader and RecursiveCharacterTextSplitter for optimized document chunking.

Vector Search: Powered by FAISS and text-embedding-3-small for high-speed, relevant context retrieval.

Advanced Retrieval (MMR): Implements Maximal Marginal Relevance (MMR) to ensure retrieved information is both relevant and diverse, avoiding redundant answers.

Structured Output: Custom prompt engineering ensures responses are delivered in concise, actionable bullet points.

Tech Stack
Framework: LangChain (LCEL - LangChain Expression Language)

Frontend: Streamlit UI 

Vector Database: FAISS

Embeddings: OpenAI text-embedding-3-small

LLM: OpenAI GPT-5-nano / GPT-3.5-Turbo

Project Architecture: 
The system follows a classic RAG pipeline:
Ingestion: PDF uploaded -> Chunked (1000 chars, 200 overlap).
Indexing: Chunks embedded and stored in a local FAISS vector store.
Retrieval: Query -> MMR Search (k=4) 
Context chunks.Generation: Context + Prompt -> RunnablePassthrough Bulleted Response.



