import os
import streamlit as st
import pandas as pd
import json
import weaviate
import re
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Optional imports with fallbacks
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False

load_dotenv()

class CreditCardSystem:
    def __init__(self):
        self.weaviate_url = os.getenv("WEAVIATE_CLUSTER_URL")
        self.weaviate_key = os.getenv("WEAVIATE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.data_path = os.getenv("CREDIT_CARD_DATA_PATH", "")
        
        self.client = None
        self.embeddings = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.llm = None
        self.tavily = None
        self.collection_name = "CreditCards"
        
        self._init_services()
    
    def _init_services(self):
        # Initialize OpenAI
        if self.openai_key and HAS_OPENAI:
            try:
                self.llm = ChatOpenAI(api_key=self.openai_key, model="gpt-3.5-turbo", temperature=0.7)
            except:
                pass
        
        # Initialize Tavily
        if self.tavily_key and HAS_TAVILY:
            try:
                self.tavily = TavilyClient(api_key=self.tavily_key)
            except:
                pass
    
    def connect_weaviate(self):
        if not self.weaviate_url or not self.weaviate_key:
            return False
        
        try:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key=self.weaviate_key)
            )
            return True
        except Exception as e:
            st.error(f"Weaviate connection failed: {e}")
            return False
    
    def load_data(self):
        if not self.data_path or not os.path.exists(self.data_path):
            # Create sample data for demo
            return pd.DataFrame({
                'Card Name': ['Sample Cashback Card', 'Sample Travel Card'],
                'Issuer': ['Bank A', 'Bank B'],
                'Annual Fee': [0, 5000],
                'Joining Fee': [0, 1000],
                'Category': ['cashback', 'travel'],
                'Rewards Category': ['shopping,dining', 'travel,hotels'],
                'Description': ['Great for daily spending', 'Perfect for travelers']
            })
        
        try:
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                df = pd.read_excel(self.data_path)
            return df.fillna('')
        except Exception as e:
            st.error(f"Data loading failed: {e}")
            return pd.DataFrame()
    
    def setup_collection(self):
        if not self.client:
            return False
        
        try:
            # Delete existing collection
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
            
            # Create new collection
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                properties=[
                    weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="card_name", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="issuer", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="annual_fee", data_type=weaviate.classes.config.DataType.NUMBER),
                    weaviate.classes.config.Property(name="category", data_type=weaviate.classes.config.DataType.TEXT),
                ]
            )
            return True
        except Exception as e:
            st.error(f"Collection setup failed: {e}")
            return False
    
    def load_cards(self, df):
        if not self.client or df.empty:
            return False
        
        collection = self.client.collections.get(self.collection_name)
        
        try:
            with collection.batch.dynamic() as batch:
                for _, row in df.iterrows():
                    text = f"Card: {row.get('Card Name', '')} Issuer: {row.get('Issuer', '')} Fee: ₹{row.get('Annual Fee', 0)} Category: {row.get('Category', '')} Rewards: {row.get('Rewards Category', '')} Description: {row.get('Description', '')}"
                    
                    vector = self.embeddings.encode([text])[0].tolist()
                    
                    batch.add_object(
                        properties={
                            "text": text,
                            "card_name": str(row.get('Card Name', '')),
                            "issuer": str(row.get('Issuer', '')),
                            "annual_fee": float(row.get('Annual Fee', 0)),
                            "category": str(row.get('Category', ''))
                        },
                        vector=vector
                    )
            return True
        except Exception as e:
            st.error(f"Data loading failed: {e}")
            return False
    
    def search_cards(self, query: str, max_fee: Optional[float] = None):
        if not self.client:
            return []
        
        try:
            collection = self.client.collections.get(self.collection_name)
            query_vector = self.embeddings.encode([query])[0].tolist()
            
            # Build filter
            where_filter = None
            if max_fee is not None:
                where_filter = weaviate.classes.query.Filter.by_property("annual_fee").less_or_equal(max_fee)
            
            # Search
            if where_filter:
                response = collection.query.near_vector(near_vector=query_vector, where=where_filter, limit=5)
            else:
                response = collection.query.near_vector(near_vector=query_vector, limit=5)
            
            return response.objects
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []
    
    def get_general_knowledge(self, query: str):
        # Built-in knowledge base for common credit card questions
        knowledge_base = {
            "rewards": """Credit card rewards work by giving you points, cashback, or miles for every purchase. You typically earn 1-5% back depending on the category. Points can be redeemed for cash, travel, or merchandise. Higher spending categories like dining or travel often earn more rewards.""",
            
            "annual fee": """Annual fees are yearly charges for having a credit card, ranging from ₹0 to ₹50,000+. Premium cards with higher fees often offer better rewards, airport lounge access, and exclusive benefits. Consider if the benefits outweigh the fee cost.""",
            
            "interest rate": """Annual Percentage Rate (APR) is the yearly interest charged on unpaid balances. In India, rates typically range from 12-45% annually. Pay your full balance by the due date to avoid interest charges completely.""",
            
            "credit vs debit": """Credit cards let you borrow money up to a limit and pay later, building credit history. Debit cards directly access your bank account money. Credit cards offer better fraud protection and rewards but require responsible spending.""",
            
            "eligibility": """Credit card eligibility depends on age (18+), income (₹15,000+ monthly for basic cards), credit score (650+ preferred), and employment status. Students and first-time users can start with secured cards or student cards.""",
            
            "credit score": """Credit scores range from 300-900 in India. 750+ is excellent for premium cards. Improve by paying bills on time, keeping credit utilization below 30%, and maintaining old accounts. Check your score free on apps like CRED or Payme."""
        }
        
        query_lower = query.lower()
        for topic, answer in knowledge_base.items():
            if topic in query_lower:
                return answer
        
        return "I don't have specific information about that topic. Try asking about rewards, annual fees, interest rates, eligibility, or credit scores."
    
    def web_search(self, query: str):
        if not self.tavily:
            return self.get_general_knowledge(query)
        
        try:
            results = self.tavily.search(query=f"credit card {query}", max_results=3)
            if "answer" in results and results["answer"]:
                return results["answer"]
            elif "results" in results and results["results"]:
                content = results["results"][0].get("content", "")
                if content:
                    return content[:500] + "..." if len(content) > 500 else content
            # Fallback to built-in knowledge if web search returns empty
            return self.get_general_knowledge(query)
        except:
            return self.get_general_knowledge(query)
    
    def classify_query(self, query: str):
        # Simple rule-based classification
        query_lower = query.lower()
        product_keywords = ['recommend', 'suggest', 'best', 'card for', 'under', 'annual fee', 'cashback', 'travel card']
        
        if any(keyword in query_lower for keyword in product_keywords):
            return "product"
        return "general"
    
    def extract_fee_limit(self, query: str):
        # Extract fee limits from query
        patterns = [
            r'under (?:₹|rs\.?|rupees?)\s*(\d+)',
            r'below (?:₹|rs\.?|rupees?)\s*(\d+)',
            r'annual fee.*?(\d+)',
            r'fee.*?under.*?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return float(match.group(1))
        return None
    
    def generate_response(self, query: str, context: str):
        if not self.llm:
            # Better fallback response formatting
            if context and context != "None":
                return f"**Answer:** {context}"
            else:
                return "Unable to generate response. Please check your API configuration."
        
        try:
            prompt = f"""You are a helpful credit card expert. Answer this query based on the context provided.

Query: {query}
Context: {context}

Provide a clear, helpful response in a conversational tone."""
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            # Fallback if LLM fails
            if context and context != "None":
                return f"**Answer:** {context}"
            else:
                return f"I couldn't process your question properly. Error: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Credit Card Finder", page_icon="💳", layout="wide")

st.title("💳 Credit Card Finder")

# Initialize system
if "system" not in st.session_state:
    st.session_state.system = CreditCardSystem()
    st.session_state.data_loaded = False

system = st.session_state.system

# Sidebar setup
with st.sidebar:
    st.header("Setup")
    
    # Check configuration
    config_ok = bool(system.weaviate_url and system.weaviate_key)
    
    if config_ok:
        st.success("✅ Weaviate configured")
    else:
        st.error("❌ Missing Weaviate credentials")
        st.info("Set WEAVIATE_CLUSTER_URL and WEAVIATE_API_KEY")
    
    # Optional services
    if system.llm:
        st.success("✅ OpenAI available")
    else:
        st.warning("⚠️ OpenAI not configured")
    
    if system.tavily:
        st.success("✅ Web search available")
    else:
        st.warning("⚠️ Web search not configured")
    
    # Data loading
    if config_ok and not st.session_state.data_loaded:
        if st.button("Load Data"):
            with st.spinner("Setting up..."):
                if system.connect_weaviate():
                    if system.setup_collection():
                        df = system.load_data()
                        if system.load_cards(df):
                            st.session_state.data_loaded = True
                            st.success(f"✅ Loaded {len(df)} cards")
                        else:
                            st.error("Failed to load cards")
    elif st.session_state.data_loaded:
        st.success("✅ Data ready")
    
    if st.session_state.data_loaded and st.button("Reset Data"):
        st.session_state.data_loaded = False
        st.rerun()

# Main interface
if not config_ok:
    st.warning("Please configure Weaviate credentials to continue.")
    st.stop()

if not st.session_state.data_loaded:
    st.info("Click 'Load Data' in the sidebar to get started.")
    st.stop()

# Chat interface
user_input = st.text_input("Ask about credit cards:", placeholder="e.g., 'Best cashback card under ₹2000' or 'How do reward points work?'")

if user_input:
    with st.spinner("Thinking..."):
        query_type = system.classify_query(user_input)
        
        if query_type == "product":
            # Product recommendation
            fee_limit = system.extract_fee_limit(user_input)
            results = system.search_cards(user_input, fee_limit)
            
            if results:
                context = "\n".join([
                    f"• {obj.properties['card_name']} by {obj.properties['issuer']} - Annual Fee: ₹{obj.properties['annual_fee']} - Category: {obj.properties['category']}"
                    for obj in results
                ])
                context = f"Here are matching credit cards:\n{context}"
            else:
                context = "No matching cards found in the database. Try different search terms or check if data is loaded properly."
        
        else:
            # General query - get web search or built-in knowledge
            context = system.web_search(user_input)
        
        # Generate final response
        response = system.generate_response(user_input, context)
    
    st.markdown("### Response:")
    st.write(response)
    
    # Show debug info if needed
    with st.expander("Debug Info"):
        st.write(f"**Query Type:** {query_type}")
        st.write(f"**Context:** {context[:200]}..." if len(context) > 200 else context)

# Quick examples
st.subheader("Try these examples:")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Product Recommendations:**
    - Best travel credit card
    - Cashback card under 5000
    - Premium cards for high earners
    """)

with col2:
    st.markdown("""
    **General Questions:**
    - How do credit card rewards work?
    - What is annual percentage rate?
    - Credit vs debit card differences
    """)

# Footer
st.markdown("---")
st.caption("Configure environment variables: WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY, OPENAI_API_KEY (optional), TAVILY_API_KEY (optional)")
