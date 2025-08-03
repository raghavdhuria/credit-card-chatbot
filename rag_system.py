from typing import Dict, List, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from data_processor import DataProcessor
from vector_store import VectorStore
from query_router import QueryRouter
from web_search import WebSearch

class RAGSystem:
    def __init__(self, config: Dict[str, str]):
        self.openai_api_key = config.get("openai_api_key", "")
        self.tavily_api_key = config.get("tavily_api_key", "")
        self.weaviate_cluster_url = config.get("weaviate_cluster_url", "")
        self.weaviate_api_key = config.get("weaviate_api_key", "")
        self.credit_card_data_path = config.get("credit_card_data_path", "")

        self.data_processor = DataProcessor(self.credit_card_data_path)
        self.vector_store = VectorStore(self.weaviate_cluster_url, self.weaviate_api_key)
        self.query_router = QueryRouter(self.openai_api_key)
        self.web_search = WebSearch(self.tavily_api_key)

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=self.openai_api_key)

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        self.product_prompt = PromptTemplate(
            template="""
You are a credit card recommendation expert. Use the conversation history and the following credit card data to answer.

CONVERSATION HISTORY:
{chat_history}

CURRENT CONTEXT (Credit Card Data):
{context}

USER QUERY: {question}

Instructions:
- Use context to suggest credit cards with card name, issuer, fees, eligibility, and rewards.
- Personalize based on user preferences from history.
- Answer clearly and helpfully.

If uncertain, suggest consulting a financial advisor.
""",
            input_variables=["context", "chat_history", "question"]
        )
        self.general_prompt = PromptTemplate(
            template="""
You are a credit card expert assistant. Provide clear answers based on search results and conversation history.

CONVERSATION HISTORY:
{chat_history}

CURRENT CONTEXT (Search Results):
{context}

USER QUERY: {question}

Instructions:
- Explain credit card concepts clearly.
- Provide accurate, practical advice.
- Account for conflicting info transparently.

If uncertain, suggest consulting a financial advisor.
""",
            input_variables=["context", "chat_history", "question"]
        )

        self.product_chain = None
        self.general_chain = None

    def initialize(self) -> None:
        self.vector_store.connect()
        print("RAG system initialized successfully")

    def _init_retrieval_chains(self) -> None:
        if self.vector_store.vector_store is None:
            print("Vector store not initialized yet")
            return
        self.product_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.vector_store.as_retriever(search_kwargs={"k":5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.product_prompt},
            return_source_documents=True,
            chain_type="stuff"
        )
        print("Retrieval chains initialized successfully")

    def process_data(self) -> None:
        self.data_processor.load_data()
        self.data_processor.clean_data()
        docs = self.data_processor.create_documents()
        self.vector_store.create_schema()
        self.vector_store.load_documents(docs)
        self._init_retrieval_chains()
        print("Data processing and loading completed")

    def process_query(self, query: str) -> str:
        query_type, filters = self.query_router.parse_query(query)
        if query_type == "PRODUCT":
            return self._process_product_query(query, filters)
        else:
            return self._process_general_query(query)

    def _process_product_query(self, query: str, filters: Dict[str, Any]) -> str:
        if self.vector_store.vector_store is None:
            return "Please process the credit card data first via the sidebar."

        if self.product_chain is None:
            self._init_retrieval_chains()
            if self.product_chain is None:
                return "Unable to access credit card database now. Try again later."

        try:
            if filters:
                docs = self.vector_store.filter_search(query, filters)
                context_text = self._format_documents(docs)
                response = self.llm.invoke(
                    self.product_prompt.format(
                        context=context_text,
                        chat_history=self.memory.buffer,
                        question=query
                    )
                )
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response.content)
                return response.content
            else:
                response = self.product_chain({"question": query})
                return response["answer"]
        except Exception as e:
            print(f"Error processing product query: {e}")
            return "Sorry, an error occurred during query processing."

    def _process_general_query(self, query: str) -> str:
        search_results = self.web_search.search(query)
        context_text = self._format_documents(search_results)
        response = self.llm.invoke(
            self.general_prompt.format(
                context=context_text,
                chat_history=self.memory.buffer,
                question=query
            )
        )
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)
        return response.content

    def _format_documents(self, documents: List[Document]) -> str:
        if not documents:
            return "No relevant information found."

        formatted = []
        for i, doc in enumerate(documents):
            if doc.metadata.get("source") == "web_search_answer":
                formatted.append(f"GENERATED ANSWER: {doc.page_content}")
            elif doc.metadata.get("source") == "web_search":
                title = doc.metadata.get("title", "Unknown Title")
                url = doc.metadata.get("url", "Unknown URL")
                formatted.append(f"SOURCE {i+1}: {title}\nURL: {url}\n{doc.page_content}")
            else:
                card_name = doc.metadata.get("card_name", "Unknown Card")
                issuer = doc.metadata.get("issuer", "Unknown Issuer")
                annual_fee = doc.metadata.get("annual_fee", "N/A")
                joining_fee = doc.metadata.get("joining_fee", "N/A")
                category = ', '.join(doc.metadata.get("category", [])) if isinstance(doc.metadata.get("category"), list) else doc.metadata.get("category", "N/A")
                rewards = ', '.join(doc.metadata.get("rewards_category", [])) if isinstance(doc.metadata.get("rewards_category"), list) else doc.metadata.get("rewards_category", "")
                eligibility = ', '.join(doc.metadata.get("eligibility", [])) if isinstance(doc.metadata.get("eligibility"), list) else doc.metadata.get("eligibility", "")
                description = doc.metadata.get("description", "")

                fmt = (
                    f"CARD {i+1}: {card_name}\n"
                    f"ISSUER: {issuer}\n"
                    f"ANNUAL FEE: ₹{annual_fee}\n"
                    f"JOINING FEE: ₹{joining_fee}\n"
                    f"CATEGORY: {category}\n"
                    f"REWARDS: {rewards}\n"
                    f"ELIGIBILITY: {eligibility}\n"
                    f"DESCRIPTION: {description}\n"
                )
                formatted.append(fmt)
        return "\n\n".join(formatted)

    def reset_conversation(self) -> None:
        self.memory.clear()
        print("Conversation history cleared")
