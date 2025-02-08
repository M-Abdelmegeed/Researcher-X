import re
import ast
import requests
from bs4 import BeautifulSoup
import fitz
from io import BytesIO

def extract_json_array(text: str):
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        extracted_array = match.group()
        try:
            return ast.literal_eval(extracted_array)
        except (SyntaxError, ValueError):
            return []
    return []

def document_loader(url):
    HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        if url.lower().endswith(".pdf") or "pdf" in response.headers["Content-Type"]:  
            pdf_document = fitz.open(stream=BytesIO(response.content), filetype="pdf")
            text_content = "\n".join([page.get_text("text") for page in pdf_document])
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text(separator=" ", strip=True)
            print(text_content)
            print('#'*120)

        return text_content

    except requests.exceptions.HTTPError as e:
        return f"HTTP Error {e.response.status_code}: {e.response.reason}"
    except requests.exceptions.RequestException as e:
        return f"Request Error: {e}"
    except Exception as e:
        return f"Error processing {url}: {e}"
