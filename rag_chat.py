from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from llm import load_llm

loader = PyPDFLoader("Career_Guide_RAG.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="career_guide"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

llm = load_llm()

prompt = PromptTemplate(
    template="""
        You are an AI career coach.

        Use the context below as your PRIMARY source of truth.
        Do NOT hallucinate or contradict the context.

        Context:
        {context}

        Chat History:
        {chat_history}

        Question:
        {question}

        Answer:
    """,
    input_variables=["context", "chat_history", "question"]
)

if __name__ == "__main__":
    print("📘 AI Career Coach (type 'exit' to quit)\n")

    chat_history = []

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        docs = retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        formatted_prompt = prompt.format(
            context=context,
            chat_history="\n".join(chat_history),
            question=question
        )

        response = llm.invoke(formatted_prompt)

        answer = response.content if hasattr(response, "content") else str(response)

        print("\nAI:", answer)
        print("-" * 60)

        chat_history.append(f"User: {question}")
        chat_history.append(f"AI: {answer}")
