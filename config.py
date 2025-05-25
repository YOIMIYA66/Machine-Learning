# config.py
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Baidu AI Studio API Configuration
AI_STUDIO_API_KEY = os.getenv("AI_STUDIO_API_KEY")
# API service domain specified in the documentation
AI_STUDIO_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"

# Large Language Model ID (choose according to your documentation and needs)
# Examples: "ernie-3.5-8k", "ernie-4.0-8k", "ernie-speed-8k", etc.
BAIDU_LLM_MODEL_NAME = "ernie-4.5-turbo-128k"

# Embedding Model ID (choose according to your documentation and needs)
# Examples: "embedding-v1", "bge-large-zh"
BAIDU_EMBEDDING_MODEL_NAME = "bge-large-zh"

# Knowledge base path
KNOWLEDGE_BASE_DIR = "knowledge_base"

# ChromaDB persistence storage path
CHROMA_PERSIST_DIR = "chroma_db"

# JSON file loading configuration (Very important - adjust according to your JSON file structure)
# This jq_schema is used to extract text content for vectorization from JSON files.
# Example:
# If JSON is: [{"title": "Case A", "text_body": "Details A"}, {"title": "Case B", "text_body": "Details B"}]
# Then jq_schema can be: '.[] | .text_body'
# Or '.[] | "Title: " + .title + "\nBody: " + .text_body'
# If JSON is: {"documents": [{"passage": "Text 1"}, {"passage": "Text 2"}]}
# Then jq_schema can be: '.documents[].passage'
# If your `divorce_litigation_texts.json` file has objects each with a 'sentence' field containing the main text:
# IMPORTANT: Please verify or modify this jq_schema according to your actual JSON file structure
# (e.g., `divorce_litigation_texts.json`).
# The current '.[] | .sentence' assumes the file is a list of objects, each having a 'sentence' field.
JSON_JQ_SCHEMA = '.[] | .sentence'

# Text splitting parameters
CHUNK_SIZE = 450
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)

# Batch size for Embedding API calls (embedding-v1 API limits to a maximum of 16 inputs per call)
EMBEDDING_BATCH_SIZE = 16