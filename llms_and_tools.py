from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import os



load_dotenv()
def get_google_search_tool():
    return GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPAPI_API_KEY"])

def get_llama_llm(temperature=0):
    return ChatGroq(temperature=temperature, model_name="llama-3.3-70b-versatile")

def get_deepseek_llm(temperature=0):
    return ChatGroq(temperature=temperature, model_name="deepseek-r1-distill-llama-70b")

def get_gemini_llm(temperature=0):
    return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.0-flash")