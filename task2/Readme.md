Task 2: Legal Research Chatbot
Overview
In this task, a legal research chatbot is implemented using semantic search and Retrieval-Augmented Generation (RAG). The system takes legal documents, extracts relevant information using vector embeddings, and generates answers to user questions based on the stored legal content.

The chatbot uses the Sentence-Transformer model for generating sentence embeddings and a Hugging Face BART model to generate answers.

Steps Involved:

Preprocess PDFs:
Used PyPDF2 and OCR techniques (via pytesseract and pdf2image) to extract text from both text-based and scanned PDFs.

Generate Embeddings:
Used the Sentence-Transformer model (all-MiniLM-L6-v2) to generate vector embeddings for the legal documents.

Create a Chatbot Interface:
Built a Streamlit app where users can input legal queries.
Used FAISS (Facebook AI Similarity Search) to perform efficient semantic search based on query vectors and retrieve relevant documents.

Answer Generation:
Combine the retrieved documents with the query and pass them to the BART model for text generation.
The system generates and returns an answer to the user's query.
