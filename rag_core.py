# rag_core.py
import json
import logging # Added
import os
import shutil # Added for get_vector_store
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter

from baidu_llm import BaiduErnieEmbeddings, BaiduErnieLLM
from config import (CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE,
                    JSON_JQ_SCHEMA, KNOWLEDGE_BASE_DIR)

# Initialize logger
logger = logging.getLogger(__name__)

# ---- Global variables for caching, to avoid repeated loading ----
_VECTOR_STORE: Optional[Chroma] = None
_QA_CHAIN: Optional[RetrievalQA] = None

# Maximum retries for vector store creation
MAX_VECTOR_STORE_RETRIES = 2


def load_and_parse_custom_json(
    filepath: str, jq_schema_hint_param: str
) -> List[Document]:
    """
    Loads and parses a custom JSON file using a JQ schema.

    Args:
        filepath: Path to the JSON file.
        jq_schema_hint_param: The JQ schema to apply for extracting data.

    Returns:
        A list of Document objects extracted from the JSON.
    """
    docs: List[Document] = []
    logger.debug(f"Entering load_and_parse_custom_json for {filepath} using JQ schema: '{jq_schema_hint_param}'")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        import jq # Import jq here as it's an optional dependency for this specific loader

        jq_program = jq.compile(jq_schema_hint_param)
        extracted_items = jq_program.input(json_data).all()
        
        item_index = 0
        for item_content in extracted_items:
            page_content_to_use = ""
            item_metadata = {}

            if isinstance(item_content, str):
                page_content_to_use = item_content
            elif isinstance(item_content, dict):
                if 'text' in item_content and isinstance(item_content['text'], str):
                    page_content_to_use = item_content['text']
                    for key, value in item_content.items():
                        if key != 'text':
                            item_metadata[key] = value
                else:
                    for value in item_content.values(): # Fallback
                        if isinstance(value, str):
                            page_content_to_use = value
                            break
                    if not page_content_to_use:
                        logger.warning(f"JQ extracted an object, but no 'text' or other string field found. Item: {item_content}")
                        continue 
            else:
                page_content_to_use = str(item_content)
                logger.debug(f"JQ extracted an item of type {type(item_content)}, converting to string. Item: {page_content_to_use[:100]}...")

            if page_content_to_use:
                metadata = {"source_filename": os.path.basename(filepath), "jq_result_index": item_index}
                metadata.update(item_metadata)
                
                if 'labels' in metadata and isinstance(metadata['labels'], list):
                    metadata['labels'] = ", ".join(metadata['labels']) if metadata['labels'] else "None"
                elif 'labels' in metadata and not isinstance(metadata['labels'], str):
                    metadata['labels'] = str(metadata['labels'])

                docs.append(Document(page_content=page_content_to_use, metadata=metadata))
                item_index += 1
            else:
                logger.debug(f"Extracted item did not yield page content. Item: {item_content}")

        if not docs:
            logger.warning(f"Failed to extract any documents from JSON file {filepath} using JQ schema.")
        else:
            logger.info(f"Successfully extracted {len(docs)} documents from {filepath} using JQ schema.")

    except json.JSONDecodeError as e_json_load:
        logger.error(f"Could not decode JSON from file {filepath}. Error: {e_json_load}")
    except ImportError:
        logger.error("The 'jq' library is required for advanced JSON processing but is not installed. Please run 'pip install jq'.")
    except Exception as e:
        logger.error(f"Error processing JSON file {filepath} with JQ: {e}")
    return docs

# --- CSV Summary Helper Functions ---
def _create_csv_general_description_doc(df: pd.DataFrame, filename: str) -> Document:
    description = (
        f"Summary information for file '{filename}':\n"
        f"This file is a CSV (Comma Separated Values) file.\n"
        f"It has a total of {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Column names include: {', '.join(df.columns.tolist())}.\n"
    )
    return Document(
        page_content=description,
        metadata={"source_filename": filename, "content_type": "csv_description"}
    )

def _create_csv_numerical_summary_doc(df: pd.DataFrame, filename: str) -> Optional[Document]:
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        return None
    
    summary_stats_text = f"\nStatistical summary of numerical columns in file '{filename}':\n"
    for col in numeric_cols:
        stats = df[col].describe()
        summary_stats_text += (
            f"Column '{col}': "
            f"Mean: {stats.get('mean', 'N/A'):.2f}, "
            f"Std: {stats.get('std', 'N/A'):.2f}, "
            f"Min: {stats.get('min', 'N/A'):.2f}, "
            f"25%: {stats.get('25%', 'N/A'):.2f}, "
            f"Median: {stats.get('50%', 'N/A'):.2f}, "
            f"75%: {stats.get('75%', 'N/A'):.2f}, "
            f"Max: {stats.get('max', 'N/A'):.2f}\n"
        )
    return Document(
        page_content=summary_stats_text,
        metadata={"source_filename": filename, "content_type": "csv_numerical_summary"}
    )

def _create_csv_categorical_summary_doc(df: pd.DataFrame, filename: str) -> Optional[Document]:
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    if non_numeric_cols.empty:
        return None

    categorical_summary_text = f"\nSummary of non-numerical columns in file '{filename}':\n"
    for col in non_numeric_cols:
        unique_values = df[col].unique()
        if len(unique_values) < 15:
            sample_unique_values = ', '.join(map(str, unique_values[:5]))
            ellipsis = '...' if len(unique_values) > 5 else ''
            categorical_summary_text += (
                f"Column '{col}' unique value examples: "
                f"{sample_unique_values}{ellipsis}\n"
            )
        else:
            categorical_summary_text += f"Column '{col}' has {len(unique_values)} unique values.\n"
    return Document(
        page_content=categorical_summary_text,
        metadata={"source_filename": filename, "content_type": "csv_categorical_summary"}
    )

def _create_csv_sample_rows_doc(df: pd.DataFrame, filename: str, max_rows_for_summary: int) -> Optional[Document]:
    num_sample_rows = min(max_rows_for_summary // 2, len(df) // 2, 5)
    if num_sample_rows <= 0:
        return None

    sample_data_text = f"\nSample data rows from file '{filename}':\n"
    for i, row in df.head(num_sample_rows).iterrows():
        row_values = [f"{col_name} is '{row[col_name]}'" for col_name in df.columns]
        row_desc = f"Row {i + 1} data: {', '.join(row_values)}.\n"
        sample_data_text += row_desc

    if len(df) > num_sample_rows * 2:
        sample_data_text += "...\n"
        for i, row in df.tail(num_sample_rows).iterrows():
            row_values = [f"{col_name} is '{row[col_name]}'" for col_name in df.columns]
            row_idx_in_tail = len(df) - num_sample_rows + i + 1 - (len(df) - num_sample_rows) # Relative to tail start
            row_desc = f"Row {len(df) - num_sample_rows + row_idx_in_tail} data: {', '.join(row_values)}.\n"
            sample_data_text += row_desc
            
    return Document(
        page_content=sample_data_text,
        metadata={"source_filename": filename, "content_type": "csv_row_samples"}
    )

def generate_csv_summary_documents(
    filepath: str, max_rows_for_summary: int = 20
) -> List[Document]:
    """
    Generates descriptive Document objects summarizing a CSV file.

    Args:
        filepath: Path to the CSV file.
        max_rows_for_summary: Maximum number of sample rows to include in the summary.

    Returns:
        A list of Document objects containing summaries of the CSV.
    """
    docs: List[Document] = []
    filename = os.path.basename(filepath)
    try:
        df = pd.read_csv(
            filepath, keep_default_na=False,
            na_values=['NaN', 'N/A', 'NA', 'nan', 'null']
        )
        
        general_desc_doc = _create_csv_general_description_doc(df, filename)
        if general_desc_doc: docs.append(general_desc_doc)
        
        numerical_summary_doc = _create_csv_numerical_summary_doc(df, filename)
        if numerical_summary_doc: docs.append(numerical_summary_doc)
            
        categorical_summary_doc = _create_csv_categorical_summary_doc(df, filename)
        if categorical_summary_doc: docs.append(categorical_summary_doc)
            
        sample_rows_doc = _create_csv_sample_rows_doc(df, filename, max_rows_for_summary)
        if sample_rows_doc: docs.append(sample_rows_doc)

        logger.info(f"Generated {len(docs)} descriptive documents for CSV file {filename}.")
    except FileNotFoundError:
        logger.error(f"CSV file not found at {filepath}")
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file {filepath} is empty.")
    except Exception as e:
        logger.error(f"Error processing CSV file {filepath}: {e}", exc_info=True)
    return docs


def load_documents_from_kb() -> List[Document]:
    """
    Loads documents from the knowledge base directory.
    Supports PDF, DOCX, XLSX/XLS, JSON, TXT, and CSV (summary) files.
    """
    loaded_docs: List[Document] = []
    logger.info("Starting to load documents from the knowledge base...") # Changed
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        logger.error(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' does not exist.") # Changed
        return []

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        file_extension = os.path.splitext(filename)[1].lower()
        logger.info(f"Processing file: {filename} (Extension: {file_extension})") # Changed

        try:
            docs_from_file: List[Document] = []
            loader_type = ""

            if file_extension == ".pdf":
                loader_type = "PyPDFLoader"
                logger.info(f"Loading PDF file: {filename}") # Changed
                loader = PyPDFLoader(filepath)
                docs_from_file = loader.load()
            elif file_extension in [".docx", ".doc"]:
                loader_type = "UnstructuredWordDocumentLoader"
                logger.info(f"Loading Word document: {filename}") # Changed
                loader = UnstructuredWordDocumentLoader(filepath, mode="elements")
                docs_from_file = loader.load()
            elif file_extension in [".xlsx", ".xls"]:
                loader_type = "UnstructuredExcelLoader"
                logger.info(f"Attempting to load Excel file: {filename} (using UnstructuredExcelLoader)") # Changed
                loader = UnstructuredExcelLoader(filepath, mode="elements")
                docs_from_file = loader.load()
            elif file_extension == ".json":
                loader_type = "CustomJSONLoader"
                logger.info(f"Processing JSON file with custom loader: {filename}") # Changed
                docs_from_file = load_and_parse_custom_json(filepath, JSON_JQ_SCHEMA)
            elif file_extension == ".txt":
                loader_type = "TextLoader"
                logger.info(f"Loading TXT file: {filename}") # Changed
                loader = TextLoader(filepath, encoding='utf-8')
                docs_from_file = loader.load()
            elif file_extension == ".csv":
                loader_type = "CSVDescriber"
                logger.info(f"Generating descriptive documents for CSV file: {filename}") # Changed
                docs_from_file = generate_csv_summary_documents(filepath)
            else:
                logger.info(f"Skipping unsupported file type: {filename}") # Changed
                continue

            for doc in docs_from_file:
                doc.metadata['source_filename'] = filename # Standardized metadata key
                # doc.metadata['source_filepath'] = filepath # Optional: if full path is needed
            
            loaded_docs.extend(docs_from_file)
            logger.info( # Changed
                f"Successfully loaded {len(docs_from_file)} document(s)/element(s) "
                f"from {filename} using {loader_type}."
            )

        except Exception as e:
            logger.error(f"Error loading document {filename}: {e}", exc_info=True) # Changed, added exc_info
            continue

    if not loaded_docs:
        logger.warning( # Changed
            "No documents were loaded. "
            "Please check the knowledge base directory and file formats/content."
        )
    else:
        logger.info(f"Total documents loaded: {len(loaded_docs)}") # Changed

    return loaded_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks.
    """
    if not documents:
        logger.info("No documents provided to split.") # Changed
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} text chunks.") # Changed
    return chunks


def get_vector_store(force_recreate: bool = False, retry_count: int = 0) -> Optional[Chroma]: # Added retry_count
    """
    Gets the vector store, loading from disk if available or creating a new one.
    Uses a global variable `_VECTOR_STORE` for in-memory caching.
    Includes a retry mechanism for creation failures.
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is not None and not force_recreate:
        logger.info("Returning existing vector database from memory.") # Changed
        return _VECTOR_STORE

    if retry_count > MAX_VECTOR_STORE_RETRIES:
        logger.error("Maximum retries reached for getting vector store. Aborting.")
        return None

    embeddings = BaiduErnieEmbeddings()
    chroma_db_exists = (
        os.path.exists(CHROMA_PERSIST_DIR) and
        os.path.isdir(CHROMA_PERSIST_DIR) and
        len(os.listdir(CHROMA_PERSIST_DIR)) > 0
    )

    if chroma_db_exists and not force_recreate:
        logger.info(f"Loading existing vector database from disk: {CHROMA_PERSIST_DIR}") # Changed
        try:
            _VECTOR_STORE = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
            _VECTOR_STORE.similarity_search("test query", k=1)
            logger.info("Vector database loaded successfully and passed test query.") # Changed
        except Exception as e:
            logger.warning( # Changed
                f"Failed to load vector database from disk: {e}. "
                f"Attempting to create a new one (retry {retry_count + 1})."
            )
            return get_vector_store(force_recreate=True, retry_count=retry_count + 1) # Incremented retry_count
    else:
        if force_recreate and chroma_db_exists:
            logger.info( # Changed
                f"Force recreate: Deleting old Chroma database at {CHROMA_PERSIST_DIR}"
            )
            shutil.rmtree(CHROMA_PERSIST_DIR) # Moved import to top
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        logger.info("Creating new vector database...") # Changed
        documents = load_documents_from_kb()
        if not documents:
            logger.error("No documents found in knowledge base. Cannot create vector database.") # Changed
            return None

        chunks = split_documents(documents)
        if not chunks:
            logger.error("No chunks produced after document splitting. Cannot create vector database.") # Changed
            return None

        try:
            logger.info("Filtering complex metadata...") # Changed
            chunks_for_chroma = filter_complex_metadata(chunks)
            logger.info( # Changed
                f"Metadata filtering complete. Processed {len(chunks_for_chroma)} "
                f"chunks for Chroma."
            )
        except Exception as e:
            logger.warning( # Changed
                f"Error filtering complex metadata: {e}. "
                f"Attempting to use original chunks."
            )
            chunks_for_chroma = chunks

        logger.info( # Changed
            f"Embedding {len(chunks_for_chroma)} text chunks and creating Chroma database. "
            f"This may take a long time..."
        )
        try:
            _VECTOR_STORE = Chroma.from_documents(
                documents=chunks_for_chroma,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            logger.info( # Changed
                f"New vector database created successfully and persisted to: {CHROMA_PERSIST_DIR}"
            )
        except Exception as e:
            logger.error(f"Critical error creating vector database: {e}", exc_info=True) # Changed
            _VECTOR_STORE = None
            if retry_count < MAX_VECTOR_STORE_RETRIES:
                logger.info(f"Retrying vector store creation (attempt {retry_count + 1}/{MAX_VECTOR_STORE_RETRIES})...")
                return get_vector_store(force_recreate=True, retry_count=retry_count + 1)
            else:
                logger.error("Max retries reached for vector store creation after critical error.")
                return None
    return _VECTOR_STORE


def get_qa_chain(force_recreate_vs: bool = False) -> Optional[RetrievalQA]:
    """
    Gets the QA chain, creating it if necessary.
    Uses global variables `_QA_CHAIN` and `_VECTOR_STORE` for caching.
    """
    global _QA_CHAIN, _VECTOR_STORE
    if _QA_CHAIN is not None and not force_recreate_vs:
        logger.info("Returning existing QA chain from memory.") # Changed
        return _QA_CHAIN

    vector_store = get_vector_store(force_recreate=force_recreate_vs)
    if vector_store is None:
        logger.error("Cannot get vector database, so QA chain cannot be created.") # Changed
        _QA_CHAIN = None 
        return None

    llm = BaiduErnieLLM()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    try:
        _QA_CHAIN = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
        )
        logger.info("QA chain created successfully.") # Changed
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}", exc_info=True) # Changed
        _QA_CHAIN = None 
    return _QA_CHAIN


def query_rag(question: str) -> Dict[str, Any]:
    """
    Queries the RAG system with a given question.
    """
    global _QA_CHAIN 
    logger.info(f"Received query: {question}") # Changed
    if not question or not isinstance(question, str) or not question.strip():
        return {"answer": "Please enter a valid question.", "source_documents": []}

    qa_chain_instance = get_qa_chain()
    if qa_chain_instance is None:
        # Changed
        logger.error("RAG QA chain not initialized successfully. Cannot answer question.")
        return {
            "answer": "Error: RAG QA chain not initialized successfully. Cannot answer question.",
            "source_documents": []
        }

    try:
        logger.info("Invoking QA chain for query...") # Changed
        result = qa_chain_instance.invoke({"query": question})
        answer = result.get("result", "Could not find a definitive answer.")
        source_docs_raw = result.get("source_documents", [])

        vector_store_instance = get_vector_store() 
        if vector_store_instance:
            logger.debug(f"Directly retrieving content related to '{question}' from vector_store") # Changed
            try:
                retrieved_docs_direct = vector_store_instance.similarity_search_with_score(
                    question, k=5
                )
                if retrieved_docs_direct:
                    logger.debug( # Changed
                        f"Top {len(retrieved_docs_direct)} "
                        f"directly retrieved documents (with scores):"
                    )
                    for i, (doc, score) in enumerate(retrieved_docs_direct):
                        source = doc.metadata.get('source_filename', 'N/A') # Use source_filename
                        page = doc.metadata.get('page', 'N/A') 
                        logger.debug( # Changed
                            f"  DOC {i + 1}, Score: {score:.4f}, Source: {source}, Page: {page}"
                        )
                        logger.debug(f"    Content (snippet): {doc.page_content[:150]}...\n") # Changed
                else:
                    logger.debug("No documents retrieved directly from vector_store.") # Changed
            except Exception as e_direct_search:
                logger.debug(f"Error during direct vector_store search: {e_direct_search}") # Changed

        formatted_sources = []
        if source_docs_raw:
            for doc_item in source_docs_raw:
                if isinstance(doc_item, Document):
                    content_snippet = doc_item.page_content
                    if len(content_snippet) > 500:
                        content_snippet = content_snippet[:500] + "..."
                    source_info = {
                        "page_content": content_snippet,
                        "metadata": doc_item.metadata # Metadata already contains source_filename
                    }
                    formatted_sources.append(source_info)

        logger.info(f"Answer: {answer}") # Changed
        if formatted_sources:
            logger.info( # Changed
                f"Reference sources (first source metadata): "
                f"{formatted_sources[0]['metadata'] if formatted_sources else 'None'}"
            )
        return {"answer": answer, "source_documents": formatted_sources}
    except Exception as e:
        logger.error(f"Error during RAG query processing: {e}", exc_info=True) # Changed
        return {"answer": f"Error processing your question: {e}", "source_documents": []}


def initialize_rag_system(force_recreate_vs: bool = False):
    """
    Initializes the RAG system by ensuring the QA chain is created.
    """
    global _QA_CHAIN 
    logger.info("Initializing RAG system...") # Changed
    try:
        qa_chain_instance = get_qa_chain(force_recreate_vs=force_recreate_vs)
        if qa_chain_instance:
            logger.info("RAG system initialized successfully.") # Changed
        else:
            logger.warning("RAG system may not have initialized completely; QA chain not created.") # Changed
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True) # Changed
        raise 


def direct_query_llm(query: str) -> Dict[str, Any]:
    """
    Directly queries the Large Language Model without using RAG.

    Args:
        query: The user's query text.

    Returns:
        A dictionary containing the answer.
    """
    logger.debug(f"Directly querying LLM with: {query[:100]}...") # Log entry
    llm = BaiduErnieLLM()

    prompt = f"""Please answer the following question directly.
Based on your existing knowledge, if you are unsure, please state so.
Keep the answer concise and accurate.

Question: {query}

Answer:"""

    answer = llm.predict(prompt)
    logger.debug(f"LLM direct response received: {answer[:100]}...") # Log exit

    return {
        "answer": answer.strip(),
        "source_documents": [],
        "is_direct_answer": True
    }