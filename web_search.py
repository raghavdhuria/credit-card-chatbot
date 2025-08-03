from tavily import TavilyClient
from typing import List
from langchain.schema import Document

class WebSearch:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> List[Document]:
        credit_card_query = f"credit card {query}"

        results = self.client.search(
            query=credit_card_query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=True,
            include_images=False
        )

        documents = []
        if "results" in results:
            for item in results["results"]:
                doc = Document(
                    page_content=item.get("content", ""),
                    metadata={"title": item.get("title", ""), "url": item.get("url", ""), "source": "web_search"}
                )
                documents.append(doc)

            if "answer" in results and results["answer"]:
                answer_doc = Document(
                    page_content=results["answer"],
                    metadata={"title": "Tavily Generated Answer", "source": "web_search_answer"}
                )
                documents.insert(0, answer_doc)

        return documents
