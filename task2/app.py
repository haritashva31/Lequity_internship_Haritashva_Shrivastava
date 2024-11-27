import streamlit as st
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the Hugging Face BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

embeddings = torch.load("task2/knowledge_base_embeddings-1.pt")

with open("task2/knowledge_base-1.txt", "r", encoding='utf-8') as file:
    corpus = file.readlines()

embeddings_np = embeddings.cpu().numpy()

index = faiss.IndexFlatL2(embeddings_np.shape[1])  
index.add(embeddings_np)

def get_relevant_documents(query, top_k=5):
    query_embedding = sentence_model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()  
    D, I = index.search(query_embedding_np, top_k)  
    relevant_documents = [corpus[i] for i in I[0]]
    return relevant_documents

def generate_answer(query, relevant_docs):
    input_text = " ".join(relevant_docs) + " " + query
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="longest")
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=50, max_length=500)
    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return answer


st.title('AskLex10 Legal Question Answering System By Haritashva')
st.subheader('Ask a legal question, and get an answer based on the knowledge base. Developed By: Haritashva')


query = st.text_input("Enter your legal question:")

if query:
    with st.spinner('Retrieving relevant documents and generating the answer...'):

        relevant_documents = get_relevant_documents(query, top_k=5)


        answer = generate_answer(query, relevant_documents)

        
        st.subheader("Answer:")
        st.write(answer)
