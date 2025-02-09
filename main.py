from langchain_core.output_parsers import StrOutputParser
from state import ResearchState
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import ast
from utils import extract_json_array, document_loader
from llms_and_tools import get_gemini_llm, get_google_search_tool, get_llama_llm, get_deepseek_llm

load_dotenv()


def main_workflow(session_id, query, message_history):
    graph = StateGraph(ResearchState)
    graph.add_node("classify_query", query_classifier)
    graph.add_node("handle_general", handle_general)
    graph.add_node("planner", planner)
    graph.add_node("google_searcher", google_searcher)
    graph.add_node("filter_and_rerank", filter_and_rerank)
    graph.add_node("research_formulator", research_formulator)
    graph.add_node("chat_summarizer", chat_summarizer)
    graph.add_node("handle_follow_up", handle_follow_up)
    graph.set_entry_point("classify_query")
    graph.add_conditional_edges(
            "classify_query",
            decide_query_type,
            {
                "research": "planner",
                "general": "handle_general",
                "follow-up": "handle_follow_up"
            },
    )
    graph.add_edge("planner", "google_searcher")
    graph.add_edge("google_searcher", "filter_and_rerank")
    graph.add_edge("filter_and_rerank", "research_formulator")
    graph.add_edge("research_formulator", "chat_summarizer")
    graph.add_edge("handle_general", "chat_summarizer")
    graph.add_edge("handle_follow_up", END)
    graph.add_edge("chat_summarizer", END)

    initial_state = ResearchState(
        query=query,
        session_id=session_id,
        retrieved_docs=[],
        classification="",
        chat_history=message_history.messages,
        chat_summary="",
        response="",
        generated_queries=[],
        reranked_docs = "",
        research_result=""
    )

    app = graph.compile()
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Node '{key}':")
        print("\n---\n")

    return value["response"]

def query_classifier(state: ResearchState) -> ResearchState:
    print("---CLASSIFY QUESTION---")
    question = state["query"]
    chat_summary = state["chat_summary"]
    print("question: ", question)

    prompt = PromptTemplate(
        template="""
        You are an AI research assistant called Researcher-X.  
        Your task is to classify the **user's query** into one of three categories:  
        - **'research'**: A new request requiring external research.  
        - **'follow-up'**: A continuation of a previous research-related conversation.  
        - **'general'**: A casual or unrelated question.  

        **Classification Rules:**  
        - If the query requests **new information that requires external research**, classify it as **'research'**.  
        - If the query is a **direct continuation of a previous research discussion**, classify it as **'follow-up'**.  
        - If the query is a **casual, unrelated, or general chat question**, classify it as **'general'**.  

        **Additional Context:**  
        - Use the chat summary to check whether the query relates to a past research discussion.  

        **Output Format:**  
        Return only a **single word**: **'research'**, **'follow-up'**, or **'general'**.  
        **DO NOT** include explanations or extra text.  

        **User Query:** {question}  
        **Chat Summary:** {chat_summary}  
        """,
        input_variables=["question", "chat_summary"],
    )

    llm = get_llama_llm()

    classification_chain = prompt | llm | StrOutputParser()
    classification = classification_chain.invoke({"question": question, "chat_summary": chat_summary}).strip().lower()
    state["classification"] = classification

    return state

def handle_follow_up(state: ResearchState) -> ResearchState:
    print("---HANDLING FOLLOW-UP REQUEST---")
    follow_up_query = state["query"]
    chat_summary = state["chat_summary"]
    
    prompt = PromptTemplate(
        template="""You are an AI research assistant called Researcher-X.  
            The user is asking a **follow-up** question based on previous research.  

            **Your Task:**  
            - Use the **chat summary** as context to generate a relevant response.  
            - If the question asks for clarification, summarize the most important details.  
            - If the question asks for **more research**, indicate that new research is required.  
            - Keep the response **concise and informative**.  

            **User Follow-Up Query:** {follow_up_query}  
            **Chat Summary:** {chat_summary}  

            **Response:**  
        """,
        input_variables=["follow_up_query", "chat_summary"],
    )

    llm = get_gemini_llm()

    response_chain = prompt | llm | StrOutputParser()
    follow_up_response = response_chain.invoke({"follow_up_query": follow_up_query, "chat_summary": chat_summary})

    state["follow_up_response"] = follow_up_response
    state["response"] = state["follow_up_response"]
    return state

def decide_query_type(state):
    classification = state["classification"]
    if classification == "research":
        return "research"
    elif classification == "follow-up":
        return "follow-up"
    else:
        return "general"

def handle_general(state: ResearchState) -> ResearchState:
    print("---HANDLE GENERAL---")
    question = state["query"]
    chat_summary = state["chat_summary"]
    prompt_template = """
        You are an AI-Agentic researcher called Researcher-X.
        Respond to the following question: {question}.
        If the chat summary is empty to not bring it up.
        Check the chat summary for more context: {chat_summary}.
        """
    prompt = PromptTemplate(input_variables=["chat_summary", "question"], template=prompt_template)

    llm = get_llama_llm()

    general_chain = prompt | llm | StrOutputParser()
    generation = general_chain.invoke({"question": question, "chat_summary": chat_summary})
    state["response"] = generation
    return state

def planner(state: ResearchState) -> ResearchState:
    print("---PLAN QUERIES---")
    prompt_template = """Given the following chat history: {chat_summary},  
        and the user's query: {query},  
        generate **exactly five** well-structured search queries to retrieve **relevant and high-quality** information from the web.  
        These queries should **expand on key aspects** of the user's topic to provide a comprehensive research scope.  

        **Guidelines:**  
        - Ensure diversity in the queries to cover different perspectives.  
        - Focus on reputable sources and precise wording.  
        - Do not include redundant or overly similar queries.  

        **Output:**  
        Return a Python list containing the five generated queries as strings, **without any additional text or explanations**.  
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "chat_summary"],
    )

    llm = get_llama_llm()
    planner_chain = prompt | llm |StrOutputParser()
    state["generated_queries"] = planner_chain.invoke({"query": state["query"], "chat_summary": state['chat_summary']})
    print("Generated Queries:", state['generated_queries'])
    return state

def google_searcher(state: ResearchState) -> ResearchState:
    print("---GOOGLE SEARCH---")
    queries = state["generated_queries"]
    if isinstance(queries, str):
        try:
            queries = ast.literal_eval(queries)
        except (ValueError, SyntaxError):
            queries = [queries]  # Fallback
    state["generated_queries"] = queries
    print(queries)
    # print("First query", queries[0])
    serper_tool = get_google_search_tool()
    for query in queries:
        results = serper_tool.results(query)
        state["retrieved_docs"].append(results)
        print(results)
    print("Retrieved Docs:", state["retrieved_docs"])
    return state

def filter_and_rerank(state: ResearchState) -> ResearchState:
    print("---FILTER AND RERANK DOCUMENTS---")
    
    prompt_template = """You are an AI research assistant tasked with filtering and reranking search results.  
        Based on the following chat history and user query, evaluate the relevance and impact of each document.  

        ### **Chat Summary:**  
        {chat_summary}  

        ### **User Query:**  
        {query}  

        ### **Documents Retrieved:**  
        {documents}  

        #### **Instructions:**  
        1. **Filter out** irrelevant, duplicate, or low-quality results.  
        2. **Rerank** the remaining documents based on their relevance to the query.  
        3. **Return results in a structured JSON format** with title, link, summary, and ranking.  

        #### **Expected JSON Response Format:**  
        
        [
            {{
                "rank": 1,
                "title": "Document Title",
                "url": "https://example.com",
                "summary": "Brief summary of why this document is relevant."
            }},
            {{
                "rank": 2,
                "title": "Document Title",
                "url": "https://example.com",
                "summary": "Brief summary of why this document is relevant."
            }}
        ]
          

        Return only the JSON array, with no additional text.  
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "chat_summary", "documents"],
    )
    gemini_llm = get_gemini_llm()
    planner_chain = prompt | gemini_llm | StrOutputParser()

    response = planner_chain.invoke({
        "query": state["query"],
        "chat_summary": state["chat_summary"],
        "documents": state["retrieved_docs"]
    })
    

    state["reranked_docs"] = response

    print("Reranked Docs:", state["reranked_docs"])
    return state


def research_formulator(state: ResearchState) -> ResearchState:
    print("---LOAD DOCS AND SUMMARIZE---")
    extracted_array = extract_json_array(state['reranked_docs'])

    # Load webpage content (Scraping)
    content_list = []
    for obj in extracted_array:
        if "url" in obj:
            url = obj["url"]
            text_content = document_loader(url)
            if text_content:
                content_list.append({url: text_content})

    aggregated_content = "\n\n".join(str(content_list))

    prompt_template = """
    You are an AI research assistant. Your task is to generate a **comprehensive research report** based on the user's query, extracting the most relevant information from the provided text.
    If the user has any preferences for the output in terms of structure or format you MUST conform to it.
    The following structure is to guide you when you're writing the report.
    ---

    ## **Research Report**
    ### **1. Introduction**
    - Provide an overview of the topic based on the user's query.
    - Define key terms or concepts if necessary.

    ### **2. Background & Context**
    - Explain relevant background information to set the stage for the research.
    - Include historical context, key developments, or industry trends.

    ### **3. Key Insights & Findings**
    - Summarize the most **important facts, statistics, and arguments** related to the query.
    - Highlight **different perspectives** if applicable.
    - Use bullet points or subheadings to structure the findings.

    ### **4. Analysis & Discussion**
    - Provide an in-depth analysis of the findings.
    - Compare different viewpoints and discuss their implications.
    - Explain any contradictions or limitations in the sources.

    ### **5. Conclusion**
    - Summarize key takeaways and insights.
    - Provide potential recommendations or next steps if applicable.

    ### **6. References**
    - List the sources used in markdown format, linking to the original documents.
    - Use `[Title](URL)` format for proper markdown citation.

    ---

    ### **Chat Summary:**  
    {chat_summary}

    ### **User Query:**  
    {query}

    ### **Extracted Documents Content:**  
    {content}

    ---

    #### **Instructions:**  
    1. Write a **well-structured research report** following the format above.  
    2. Ensure the content is **detailed, well-organized, and professional**.  
    3. **Cite sources** using markdown links `[Title](URL)`.  
    4. If multiple sources mention the same information, **synthesize the information** concisely.  

    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "chat_summary", "content"],
    )


    gemini_llm = get_gemini_llm()
    research_formulator_chain = prompt | gemini_llm | StrOutputParser()

    state["research_result"] = research_formulator_chain.invoke({
        "query": state["query"], 
        "chat_summary": state["chat_summary"], 
        "content": aggregated_content
    })

    print("Summarization Completed:", state["research_result"])
    state["response"] = state["research_result"] #TODO Remove
    return state

def chat_summarizer(state: ResearchState) -> ResearchState:
    template = """
    You are an AI assistant summarizing a conversation. 
    Prioritize the user's requests, questions, data, and needs while keeping relevant AI responses.
    All user data, preferences and needs should be in the summary.
    Keep it concise but informative. If no history is given, return " ".

    ### Conversation History:
    {message_history}
    """
    prompt = PromptTemplate(input_variables=["message_history"], template=template)
    gemini_llm = get_gemini_llm()
    chain = prompt | gemini_llm | StrOutputParser()
    summary = chain.invoke({"message_history": state["chat_history"]})
    print("Summary: ", summary.strip())
    return state