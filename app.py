import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from llm import load_llm
from rank_bm25 import BM25Okapi
import re
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="AI Career Coach",
    page_icon="📘",
    layout="centered"
)

import os
import json
from langchain_core.documents import Document

def load_json_documents(folder):
    docs = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

                text = ""

                for key, value in data.items():
                    text += f"{key}:\n"

                    if isinstance(value, list):
                        for item in value:
                            text += f"- {item}\n"
                    else:
                        text += f"{value}\n"

                    text += "\n"

                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file}
                    )
                )

    return docs

def web_search(query):
    results_text = ""

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)

        for r in results:
            results_text += r["body"] + "\n"

    return results_text

@st.cache_resource
def load_rag_pipeline():
    # loader = PyPDFLoader("Career_Guide_RAG.pdf")
    # pdf_docs = loader.load()

    faq_docs = load_json_documents("faq_docs")
    tech_docs = load_json_documents("tech_docs")
    career_docs = load_json_documents("career_docs")
    interview_docs = load_json_documents("Interview_qa_docs")
    roadmap_docs = load_json_documents("roadmap_docs")

    # documents = (
    #     pdf_docs
    #     + faq_docs
    #     + tech_docs
    #     + career_docs
    #     + interview_docs
    #     + roadmap_docs
    # )
    documents = (
        faq_docs
        + tech_docs
        + career_docs
        + interview_docs
        + roadmap_docs
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Keyword search index
    texts = [doc.page_content for doc in chunks]
    tokenized_corpus = [re.findall(r"\w+", text.lower()) for text in texts]

    bm25 = BM25Okapi(tokenized_corpus)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="career_guide"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    llm = load_llm()

    prompt = PromptTemplate(
        template="""
        You are an AI career coach.

        Use the context below to answer the question.
        The context comes from curated career knowledge sources.

        If the context does not fully answer the question,
        you may supplement with your own knowledge.

        Context:
        {context}

        Chat History:
        {chat_history}

        Question:
        {question}

        Answer in a clear structured way.
    """,
    input_variables=["context", "chat_history", "question"]
)
    input_variables=["context", "chat_history", "question"]

    return retriever, llm, prompt, bm25, chunks


retriever, llm, prompt, bm25, chunks = load_rag_pipeline()


st.title("📘 AI Career Coach")
st.caption("Ask career-related questions...")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask a career-related question...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    docs = retriever.invoke(user_input)

    # Keyword Retrieval/ Keyword Search
    query_tokens = re.findall(r"\w+", user_input.lower())
    scores = bm25.get_scores(query_tokens)

    top_index = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:2]

    keyword_docs = [chunks[i] for i in top_index]

    docs = docs + keyword_docs # Vector + keyword results are merged
   
    context_docs = docs

    context = "\n\n".join(doc.page_content for doc in context_docs)

    if len(context) < 300:
        web_context = web_search(user_input)
        context = context + "\n\nWeb Search Results:\n" + web_context

    print("\nRETRIEVED DOCS:")
    for d in docs:
        print(d.metadata)
    
    json_docs = [d for d in docs if d.metadata.get("source","").endswith(".json")]
    other_docs = [d for d in docs if not d.metadata.get("source","").endswith(".json")]

    ordered_docs = json_docs + other_docs

    context = "\n\n".join(doc.page_content for doc in ordered_docs)  

    chat_history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.messages
    )

    formatted_prompt = prompt.format(
        context=context,
        chat_history=chat_history_text,
        question=user_input
    )
    response = llm.invoke(formatted_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
