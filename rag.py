import os

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# FAISS vector database'i yükle
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Sen teknik doküman sorularını cevaplayan bir asistansın.
            Kullanıcının sorusunu sadece sana verilen context içerisindeki
            bilgileri kullanarak cevapla.

            Context içerisinde cevap bulunmuyorsa:
            "Bu bilgi dokümanda bulunmuyor."
            şeklinde cevap ver.
            """
        ),
        (
            "human",
            """CONTEXT:{context}
             Soru:{question}""")])

# Output parser
output_parser = StrOutputParser()


# Chain
chain = prompt | model | output_parser


# Kullanıcı sorusu
question = "Elektronik kontrol birimini max calısma sıcaklığı nedir?"


# Retriever ile ilgili dokümanları getir
docs = retriever.invoke(question)


# Doküman içeriklerini context haline getir
context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# Chain'i çalıştır
answer = chain.invoke(
    {
        "context": context,
        "question": question
    }
)


print(answer)