import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# .env dosyasını yükle
load_dotenv()

# API key'i al
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("API key bulunamadı!")
    exit()


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=google_api_key
)

user_prompt = input("Sorunuzu yazın: ")

response = model.invoke(user_prompt)

print("\nGemini:")
print(response.content)