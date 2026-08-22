import os

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

def load_llm(huggingface_repo_id):
    base_llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    chat_llm = ChatHuggingFace(llm=base_llm)
    return chat_llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {input}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])
    return prompt

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

llm = load_llm(HUGGINGFACE_REPO_ID)
prompt_layout = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

document_combination_chain = create_stuff_documents_chain(llm, prompt_layout)
retriever_engine = db.as_retriever(search_kwargs={'k': 3})
qa_chain = create_retrieval_chain(retriever_engine, document_combination_chain)

user_query=input("Write Query Here: ")
response=qa_chain.invoke({'input': user_query})
print("RESULT: ", response["answer"])
print("SOURCE DOCUMENTS: ", response["context"])