# Researcher-X: Your AI Research Agent

Researcher-X is an advanced AI-powered research agent designed to provide comprehensive and accurate research on any topic. Leveraging state-of-the-art large language models (LLMs) and real-time web searches, Researcher-X can analyze, summarize, and respond to complex queries efficiently. 

Built with  tools like **LangGraph** and **Streamlit**, Researcher-X offers a user-friendly web interface for seamless interactions. To ensure continuity and context, it uses **MongoDB** to store and retrieve chat history, enabling meaningful follow-up discussions.

Whether you're exploring new topics, diving into scientific research, or seeking general information, Researcher-X is your go-to tool for personalized research assistance.

---

### Features
- **Comprehensive Research**: Combines web search and LLM-based reasoning to provide detailed, accurate information.
- **Follow-Up Handling**: Keeps track of previous interactions for enhanced contextual understanding.
- **Modular Design**: Built using LangGraph for scalable and maintainable workflows.
- **Web-Based Interface**: A sleek, interactive UI powered by Streamlit.
- **Persistent Chat History**: Stores conversations in MongoDB for continuity.

---

## Architecure

![Screenshot 2025-02-08 201559](https://github.com/user-attachments/assets/35c0f0fb-5e4b-4c7a-a428-ae2dba9a4aae)

1. **User Query Input**:
   - Users interact with Researcher-X through a web interface powered by **Streamlit**.

2. **Query Classification**:
   - The **Classify Query** component determines the nature of the query:
     - **General Queries**: Basic questions handled directly by the system, such as greetings and general questions
     - **Research Queries**: Complex questions requiring multi-step research.
     - **Follow-Up Queries**: Continuations of previous discussions related to the research topic

3. **Handling Queries**:
   - **Handle General**: Resolves basic queries and provides direct responses.
   - **Handle Follow-Up**: Pulls context from MongoDB and uses the summary memory to address follow-up questions, regarding research.
   - **Research Planner**: For research queries, the planner decomposes the question into smaller tasks or queries, to be searched for in the next step

4. **Search and Evaluation**:
   - Queries are distributed to multiple **Searcher Components**, which use the Google Serper tool to search the web using the queries that were generated by the planner.
   - Retrieved documents are **aggregated, evaluated, and re-ranked** for relevance with respect to the user's original query, and quality.

5. **Top Document Processing**:
   - The top-ranked documents are **loaded** so that they can be used for context to generate the report
   - The **Research Formulator** generates detailed Markdown-based research outputs.

6. **Research Formulator**:
   - The **Research Formulator** uses all the relevant documents and generates a report based on the user's output preference (MarkDown, etc..), and also provides the user with links for references

7. **Chat Summarizer**:
   - The **chat summarizer** summarizes the chat so that it can be used as context again and again for the reasoning agent too keep track of the on-going conversation without having to use too much of the context window and tokens.

8. **LLM Integration**:  
   - Researcher-X leverages two advanced language models:  
     - **LLaMA 70B from Groq**: Used for handling basic tasks such as responding to general queries and simple follow-ups.  
     - **Gemini 2.0 Flash**: Employed in tasks requiring a high context window, such as aggregating results and generating comprehensive outputs in the **Research Formulator** stage.  
   - This dual-model setup ensures efficient handling of both lightweight and complex queries, optimizing performance and response quality.
  
  ---

## Installation and Usage

Follow these steps to install and run Researcher-X on your local machine:

 ### 1. **Clone the Repository**  
 ```bash
 git clone https://github.com/M-Abdelmegeed/researcher-x.git
 cd researcher-x
 ```

### 2. Install Dependencies
Make sure you have Python installed on your system. Then, install the required dependencies by running:

```bash
pip install -r requirements.txt
```
### 3. Set Environment Variables
Researcher-X requires four environment variables to function properly. Below is a list of the variables and how to obtain them:

| Environment Variable | Description | How to Obtain |
|----------------------|-------------|---------------|
| `MONGODB_CONNECTION_STRING`        | MongoDB connection URI | From your MongoDB Atlas or local MongoDB setup. |
| `GOOGLE_API_KEY`     | API key for Gemini-2.0-Flash LLM | Obtain from Google AI Studio portal. (https://aistudio.google.com/app/apikey) |
| `GROQ_API_KEY`      | API key for LLaMA 70B (Groq-based deployment) | Obtain from https://console.groq.com/keys |
| `SERPAPI_API_KEY`     | API key for accessing Google Serper | Obtain from https://serper.dev/api-key |

Create a `.env` file in the root directory of the project and add the following lines:

```env
MONGODB_CONNECTION_STRING=<your_mongodb_uri>
GOOGLE_API_KEY=<your_gemini_api_key>
GROQ_API_KEY=<your_groq_api_key>
SERPAPI_API_KEY=<your_serp_api_key>
```

### 4. Run the Application
Once the dependencies are installed and the environment variables are configured, start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to access the Researcher-X web interface.

That's it! You are ready to use Researcher-X for your research needs.
