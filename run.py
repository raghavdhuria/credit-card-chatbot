import os
import subprocess
import sys
from dotenv import load_dotenv

def check_dependencies():
    required_packages = [
        "langchain", "langchain_openai", "langchain_weaviate", "langchain_community",
        "streamlit", "weaviate", "tavily", "dotenv", "pandas",
        "openpyxl", "tqdm", "numpy", "sentence_transformers"
    ]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        choice = input("Install missing packages? (y/n): ")
        if choice.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        else:
            print("Please install missing packages and rerun.")
            sys.exit(1)

def check_env_file():
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=\n")
            f.write("TAVILY_API_KEY=\n")
            f.write("WEAVIATE_CLUSTER_URL=\n")
            f.write("WEAVIATE_API_KEY=\n")
            f.write("CREDIT_CARD_DATA_PATH=Credit_Card_Data.xlsx\n")
        print("Created .env file. Please update your keys and dataset path.")
        sys.exit(1)
    else:
        load_dotenv()
        missing_vars = [v for v in ["OPENAI_API_KEY", "TAVILY_API_KEY", "WEAVIATE_CLUSTER_URL", "WEAVIATE_API_KEY", "CREDIT_CARD_DATA_PATH"] if not os.getenv(v)]
        if missing_vars:
            print(f"Missing environment variables: {missing_vars}")
            print("Please edit your .env file.")
            sys.exit(1)

def run_app():
    print("Starting Credit Card Recommendation System...")
    subprocess.call(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    check_dependencies()
    check_env_file()
    run_app()
