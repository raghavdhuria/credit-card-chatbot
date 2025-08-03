from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, Tuple, Literal
import json

class QueryRouter:
    """
    Routes queries into credit card recommendation (PRODUCT) or general advice (GENERAL) categories,
    and extracts filters such as annual fee, joining fee, and categories.
    """

    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key
        )

        self.classification_prompt = PromptTemplate(
            template="""
You are a financial assistant routing credit card and debit card queries.

Determine if a query is about:

1. PRODUCT - specific credit/debit card recommendations

2. GENERAL - questions about credit cards, features, eligibility, or advice

Reply ONLY with "PRODUCT" or "GENERAL".

Examples:

- Recommend a cashback credit card under ₹5000 -> PRODUCT

- How do credit card rewards work? -> GENERAL

Query: {query}
""",
            input_variables=["query"]
        )

        self.classification_chain = LLMChain(llm=self.llm, prompt=self.classification_prompt)

        self.filter_prompt = PromptTemplate(
            template="""
Extract filters for "annual fee", "joining fee" and "categories" from this credit/debit card query.
Return a JSON object with keys "annual_fee_max", "joining_fee_max", "categories" where applicable.
If not specified, return an empty JSON.

Examples:

- Credit cards with annual fee under 500 -> {{"annual_fee_max":500}}

- No joining fee debit cards for students -> {{"joining_fee_max":0, "categories":["student"]}}

- Best travel debit cards -> {{"categories":["travel"]}}

Query: {query}

Return JSON:
""",
            input_variables=["query"]
        )

        self.filter_chain = LLMChain(llm=self.llm, prompt=self.filter_prompt)

    def classify_query(self, query: str) -> Literal["PRODUCT", "GENERAL"]:
        res = self.classification_chain.run(query).strip().upper()
        return "PRODUCT" if res == "PRODUCT" else "GENERAL"

    def extract_filters(self, query: str) -> Dict[str, Any]:
        raw = self.filter_chain.run(query).strip()
        try:
            filters = json.loads(raw)
            # Normalize keys and values as needed
            if not isinstance(filters, dict):
                filters = {}
        except Exception:
            filters = {}
        return filters

    def parse_query(self, query: str) -> Tuple[Literal["PRODUCT", "GENERAL"], Dict[str, Any]]:
        qtype = self.classify_query(query)
        filters = {}
        if qtype == "PRODUCT":
            filters = self.extract_filters(query)
        return qtype, filters
