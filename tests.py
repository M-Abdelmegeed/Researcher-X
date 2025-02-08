# from langchain_community.tools.google_scholar import GoogleScholarQueryRun
# from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
# from dotenv import load_dotenv

# load_dotenv()

# tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
# print(tool.run("Attention is all you need"))

################################################################################################

# from langchain_community.utilities.arxiv import ArxivAPIWrapper
# arxiv = ArxivAPIWrapper(
#     top_k_results = 3,
#     ARXIV_MAX_QUERY_LENGTH = 300,
#     load_max_docs = 3,
#     load_all_available_meta = False,
#     doc_content_chars_max = 50000
# )
# result = arxiv.run("Attention & Transformers")
# print(result)

#################################################################################################

# from langchain_community.tools import DuckDuckGoSearchRun

# search = DuckDuckGoSearchRun(output_format="list")

# print(search.invoke("Obama's first name?"))

##################################################################################################

# from langchain_unstructured import UnstructuredLoader

# loader = UnstructuredLoader(web_url="https://www.liveeatlearn.com/tropical-fruits/")
# docs = loader.load()

# # Extracting only the main content
# for doc in docs:
#     print(doc.page_content)  # Only prints the page content