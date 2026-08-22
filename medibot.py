import os
import tempfile

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say that you don't know instead of making one up.
Do not provide anything that is not in the given context.

Context: {context}
Question: {input}

Start the answer directly. No small talk.
"""


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def save_uploaded_pdfs(uploaded_files):
    temp_dir = tempfile.mkdtemp(prefix="pdf_upload_")

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getvalue())

    return temp_dir


def build_vectorstore_from_pdf_dir(pdf_dir):
    loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        raise ValueError("No readable text was found in the uploaded PDF file.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    embedding_model = get_embedding_model()
    return FAISS.from_documents(text_chunks, embedding_model)


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])


def load_llm(huggingface_repo_id):
    base_llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
    )
    return ChatHuggingFace(llm=base_llm)


def create_qa_chain(vectorstore):
    llm = load_llm(HUGGINGFACE_REPO_ID)
    prompt_layout = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    document_combination_chain = create_stuff_documents_chain(llm, prompt_layout)
    retriever_engine = vectorstore.as_retriever(search_kwargs={"k": 5})
    return create_retrieval_chain(retriever_engine, document_combination_chain)


def main():
    st.title("PDF Q&A Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "vectorstore_key" not in st.session_state:
        st.session_state.vectorstore_key = None

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        upload_key = "|".join(f"{file.name}:{file.size}" for file in uploaded_files)

        if st.session_state.vectorstore is None or st.session_state.vectorstore_key != upload_key:
            try:
                with st.spinner("Processing the uploaded PDF(s)..."):
                    pdf_dir = save_uploaded_pdfs(uploaded_files)
                    st.session_state.vectorstore = build_vectorstore_from_pdf_dir(pdf_dir)
                    st.session_state.vectorstore_key = upload_key
                    st.session_state.messages = []
                st.success(f"Loaded {len(uploaded_files)} PDF file(s). You can ask questions now.")
            except Exception as exc:
                st.error(f"Unable to process the uploaded PDF: {exc}")
                st.session_state.vectorstore = None
                st.session_state.vectorstore_key = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your PDF", disabled=st.session_state.vectorstore is None)

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.vectorstore is None:
            st.warning("Please upload a PDF before asking a question.")
            st.stop()

        try:
            qa_chain = create_qa_chain(st.session_state.vectorstore)
            response = qa_chain.invoke({"input": prompt})

            result = response["answer"]
            # source_documents = response["context"]
            result_to_show = f"{result}\n"
            # result_to_show = f"{result}\n\n**Source Docs:**\n{source_documents}"

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})
        except Exception as exc:
            st.error(f"Error while generating the answer: {exc}")


if __name__ == "__main__":
    main()
