# rag_core.py
import os
import json
import pandas as pd
from typing import List, Optional, Dict, Any
import numpy as np  # 确保导入

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredWordDocumentLoader  # 添加DOCX加载器
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from config import (
    KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR, JSON_JQ_SCHEMA,
    CHUNK_SIZE, CHUNK_OVERLAP
)
from baidu_llm import BaiduErnieEmbeddings, BaiduErnieLLM

# ---- 全局变量，用于缓存，避免重复加载 ----
_VECTOR_STORE: Optional[Chroma] = None
_QA_CHAIN: Optional[RetrievalQA] = None


# --------------------------------


# rag_core.py

# rag_core.py

# rag_core.py

def load_and_parse_custom_json(filepath: str, jq_schema_hint_param: str) -> List[Document]:
    docs: List[Document] = []
    # 这个参数现在更多的是一个"概念上"的提示，我们主要关注从 list of dicts 中提取 'sentence'

    print(f"--- Debug: Entrando en load_and_parse_custom_json para {filepath} ---")
    # print(f"--- Debug: jq_schema_hint_param recibido: '{jq_schema_hint_param}' ---") # 可以保留或移除
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        open_brackets = 0
        start_index = -1
        found_segments = 0

        for i, char in enumerate(content):
            if char == '[':
                if open_brackets == 0:
                    start_index = i
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
                if open_brackets == 0 and start_index != -1:
                    found_segments += 1
                    json_array_str = content[start_index: i + 1]
                    # print(f"\n--- Debug: Segmento JSON potencial encontrado (num {found_segments}): ---")
                    # print(json_array_str[:200] + "..." if len(json_array_str) > 200 else json_array_str)

                    try:
                        json_array = json.loads(json_array_str)
                        # print(f"--- Debug: Segmento parseado con json.loads(). Tipo: {type(json_array)} ---")

                        if isinstance(json_array, list):  # <--- 只要它是列表，我们就尝试提取 sentence
                            print(
                                f"--- Debug: Segmento {found_segments} es una lista. Procesando items para extraer 'sentence'... ---")
                            items_processed_in_segment = 0
                            for item_index, item in enumerate(json_array):
                                if isinstance(item, dict) and 'sentence' in item:  # 检查每个元素是否是包含'sentence'的字典
                                    sentence_text = item.get('sentence')
                                    if isinstance(sentence_text, str):
                                        metadata = {"source": os.path.basename(filepath),
                                                    "segment_index": found_segments,
                                                    "item_index_in_segment": item_index}
                                        if 'labels' in item and isinstance(item['labels'], list):
                                            metadata['labels'] = ", ".join(item['labels']) if item['labels'] else "None"
                                        docs.append(Document(page_content=sentence_text, metadata=metadata))
                                        items_processed_in_segment += 1
                            if items_processed_in_segment == 0:
                                print(
                                    f"--- Debug: En el segmento {found_segments}, no se encontraron items válidos con 'sentence'. ---")
                            else:
                                print(
                                    f"--- Debug: En el segmento {found_segments}, se procesaron {items_processed_in_segment} items. Documentos actuales: {len(docs)} ---")
                        else:
                            print(
                                f"警告 (Debug): JSON文件 {filepath} 中的片段未解析为列表。实际类型: {type(json_array)} 内容片段: {str(json_array)[:100]}...")
                    except json.JSONDecodeError as jde:
                        print(f"解析JSON片段时发生错误 (Debug): {jde}，片段: '{json_array_str[:200]}...'")
                    except Exception as e_inner:
                        print(f"处理解析后的JSON片段时发生未知错误 (Debug): {e_inner}")
                    start_index = -1

        if found_segments == 0:
            print(
                f"--- Debug: 没有在文件 '{filepath}' 中找到任何匹配 '[...]' 的片段。检查文件是否为空或格式完全不同。 ---")

        if not docs:
            print(f"警告 (Debug): 未能从自定义JSON文件 {filepath} 中提取任何文档。")
        else:
            print(f"--- Debug: 从 {filepath} 成功提取了 {len(docs)} 个文档。 ---")

    except Exception as e:
        print(f"加载或解析自定义JSON文件 {filepath} 时发生错误 (Debug): {e}")
    return docs
def generate_csv_summary_documents(filepath: str, max_rows_for_summary: int = 20) -> List[Document]:
    # ... (代码与之前版本相同) ...
    docs: List[Document] = []
    filename = os.path.basename(filepath)
    try:
        # 处理NaN值
        df = pd.read_csv(filepath, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
        description = (
            f"关于文件 '{filename}' 的摘要信息：\n"
            f"该文件是一个CSV（逗号分隔值）文件。\n"  # 移除了具体数据类型的描述，使其更通用
            f"它总共有 {df.shape[0]} 行数据和 {df.shape[1]} 列。\n"
            f"列名包括: {', '.join(df.columns.tolist())}。\n"
        )
        docs.append(
            Document(page_content=description, metadata={"source": filename, "content_type": "csv_description"}))
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            summary_stats_text = f"\n文件 '{filename}' 中数值列的统计摘要:\n"
            for col in numeric_cols:
                stats = df[col].describe()
                summary_stats_text += (
                    f"列 '{col}': "
                    f"均值: {stats.get('mean', 'N/A'):.2f}, 标准差: {stats.get('std', 'N/A'):.2f}, "
                    f"最小值: {stats.get('min', 'N/A'):.2f}, 25分位数: {stats.get('25%', 'N/A'):.2f}, "
                    f"中位数(50%): {stats.get('50%', 'N/A'):.2f}, 75分位数: {stats.get('75%', 'N/A'):.2f}, "
                    f"最大值: {stats.get('max', 'N/A'):.2f}\n"
                )
            docs.append(Document(page_content=summary_stats_text,
                                 metadata={"source": filename, "content_type": "csv_numerical_summary"}))
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            categorical_summary_text = f"\n文件 '{filename}' 中非数值列的摘要:\n"
            for col in non_numeric_cols:
                unique_values = df[col].unique()
                if len(unique_values) < 15:
                    categorical_summary_text += f"列 '{col}' 的唯一值示例: {', '.join(map(str, unique_values[:5]))}{'...' if len(unique_values) > 5 else ''}\n"
                else:
                    categorical_summary_text += f"列 '{col}' 有 {len(unique_values)} 个唯一值。\n"
            docs.append(Document(page_content=categorical_summary_text,
                                 metadata={"source": filename, "content_type": "csv_categorical_summary"}))
        num_sample_rows = min(max_rows_for_summary // 2, len(df) // 2, 5)
        if num_sample_rows > 0:
            sample_data_text = f"\n文件 '{filename}' 的部分数据行示例:\n"
            for i, row in df.head(num_sample_rows).iterrows():
                row_desc = f"第 {i + 1} 行数据: " + ", ".join(
                    [f"{col_name}为'{row[col_name]}'" for col_name in df.columns]) + "。\n"
                sample_data_text += row_desc
            if len(df) > num_sample_rows * 2:
                sample_data_text += "...\n"
                for i, row in df.tail(num_sample_rows).iterrows():
                    row_desc = f"第 {i + 1} 行数据: " + ", ".join(
                        [f"{col_name}为'{row[col_name]}'" for col_name in df.columns]) + "。\n"
                    sample_data_text += row_desc
            docs.append(Document(page_content=sample_data_text,
                                 metadata={"source": filename, "content_type": "csv_row_samples"}))
        print(f"为CSV文件 {filename} 生成了 {len(docs)} 个描述性文档。")
    except FileNotFoundError:
        print(f"错误: CSV文件未找到于 {filepath}")
    except pd.errors.EmptyDataError:
        print(f"错误: CSV文件 {filepath} 为空。")
    except Exception as e:
        print(f"处理CSV文件 {filepath} 时发生错误: {e}")
    return docs


# --- load_documents_from_kb 函数 (修改以支持DOCX) ---
def load_documents_from_kb() -> List[Document]:
    loaded_docs: List[Document] = []
    print("开始从知识库加载文档...")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"错误: 知识库目录 '{KNOWLEDGE_BASE_DIR}' 不存在。")
        return []
    
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        file_extension = os.path.splitext(filename)[1].lower()
        try:
            if file_extension == ".pdf":
                print(f"正在加载PDF文件: {filename}")
                loader = PyPDFLoader(filepath)
                pdf_docs = loader.load()
                
                # 确保文件名被添加到元数据
                for doc in pdf_docs:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = filename
                    else:
                        # 如果已有source但只包含路径，添加文件名
                        doc.metadata['file_name'] = filename
                
                loaded_docs.extend(pdf_docs)
                print(f"成功加载PDF: {filename}, 共{len(pdf_docs)}页")
                
            elif file_extension in [".docx", ".doc"]:
                print(f"正在加载Word文档: {filename}")
                try:
                    loader = UnstructuredWordDocumentLoader(filepath, mode="elements")
                    word_docs = loader.load()
                    
                    # 确保文件名被添加到元数据
                    for doc in word_docs:
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = filename
                        else:
                            # 如果已有source但只包含路径，添加文件名
                            doc.metadata['file_name'] = filename
                    
                    loaded_docs.extend(word_docs)
                    print(f"成功加载Word文档: {filename}, 共{len(word_docs)}个元素")
                except Exception as doc_error:
                    print(f"加载Word文档 {filename} 失败: {doc_error}")
                    
            elif file_extension in [".xlsx", ".xls"]:
                print(f"正在尝试加载XLSX/XLS: {filename} (使用UnstructuredExcelLoader)")
                try:
                    loader = UnstructuredExcelLoader(filepath, mode="elements")
                    excel_docs = loader.load()
                    
                    # 确保文件名被添加到元数据
                    for doc in excel_docs:
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = filename
                        else:
                            # 如果已有source但只包含路径，添加文件名
                            doc.metadata['file_name'] = filename
                    
                    loaded_docs.extend(excel_docs)
                    print(f"成功加载XLSX/XLS: {filename}, 共{len(excel_docs)}个元素")
                except Exception as ue_error:
                    print(
                        f"使用 UnstructuredExcelLoader 加载 {filename} 失败: {ue_error}. 如果是表格数据，请考虑转为CSV格式。")
            elif file_extension == ".json":
                print(f"正在使用自定义加载器处理JSON文件: {filename}")
                custom_loaded_json_docs = load_and_parse_custom_json(filepath, JSON_JQ_SCHEMA)
                if custom_loaded_json_docs:
                    # 确保文件名被添加到元数据
                    for doc in custom_loaded_json_docs:
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = filename
                    
                    loaded_docs.extend(custom_loaded_json_docs)
                    print(f"通过自定义加载器成功从JSON文件 {filename} 加载 {len(custom_loaded_json_docs)} 个文档。")
                else:
                    print(f"警告: 自定义加载器未能从JSON文件 {filename} 中提取文档。")
            elif file_extension == ".txt":
                print(f"正在加载TXT文件: {filename}")
                loader = TextLoader(filepath, encoding='utf-8')
                txt_docs = loader.load()
                
                # 确保文件名被添加到元数据
                for doc in txt_docs:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = filename
                    else:
                        # 如果已有source但只包含路径，添加文件名
                        doc.metadata['file_name'] = filename
                
                loaded_docs.extend(txt_docs)
                print(f"成功加载TXT: {filename}, 共{len(txt_docs)}个文本块")
            elif file_extension == ".csv":
                print(f"正在为CSV文件生成描述性文档: {filename}")
                csv_summary_docs = generate_csv_summary_documents(filepath)
                if csv_summary_docs:
                    loaded_docs.extend(csv_summary_docs)
                else:
                    print(f"警告: 未能为CSV文件 {filename} 生成描述性文档。")
            else:
                print(f"跳过不支持的文件类型: {filename}")
        except Exception as e:
            print(f"加载文档 {filename} 时发生顶层错误: {e}")
            continue
            
    if not loaded_docs:
        print("警告: 未加载到任何文档。请检查知识库目录和文件格式/内容。")
    else:
        print(f"总共加载文档数量: {len(loaded_docs)}")
        
    return loaded_docs


# --- split_documents 函数 (保持不变) ---
def split_documents(documents: List[Document]) -> List[Document]:
    # ... (代码与之前版本相同) ...
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"将 {len(documents)} 个文档切分为 {len(chunks)} 个文本块。")
    return chunks


# --- get_vector_store 函数 (添加 global 声明) ---
def get_vector_store(force_recreate: bool = False) -> Optional[Chroma]:
    global _VECTOR_STORE  # <--- 添加这一行
    if _VECTOR_STORE is not None and not force_recreate:
        print("从内存返回已存在的向量数据库。")
        return _VECTOR_STORE

    embeddings = BaiduErnieEmbeddings()
    chroma_db_exists = os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0

    if chroma_db_exists and not force_recreate:
        print(f"从磁盘加载已存在的向量数据库: {CHROMA_PERSIST_DIR}")
        try:
            _VECTOR_STORE = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
            try:
                _VECTOR_STORE.similarity_search("test", k=1)
                print("向量数据库加载成功并通过测试查询。")
            except Exception as e:
                print(f"向量数据库加载后测试查询失败: {e}。可能需要重建。")
                return get_vector_store(force_recreate=True)  # 注意: 递归调用也需要global
        except Exception as e:
            print(f"从磁盘加载向量数据库失败: {e}。将尝试创建新的向量数据库。")
            return get_vector_store(force_recreate=True)  # 注意: 递归调用也需要global

    else:
        if force_recreate and chroma_db_exists:
            print(f"强制重建: 正在删除旧的Chroma数据库于 {CHROMA_PERSIST_DIR}")
            import shutil
            shutil.rmtree(CHROMA_PERSIST_DIR)
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        print("正在创建新的向量数据库...")
        documents = load_documents_from_kb()
        if not documents:
            print("错误: 知识库中未找到任何文档，无法创建向量数据库。")
            return None

        chunks = split_documents(documents)
        if not chunks:
            print("错误: 文档切分后未产生任何文本块，无法创建向量数据库。")
            return None

        try:
            print("正在过滤复杂元数据...")
            chunks_for_chroma = filter_complex_metadata(chunks)
            print(f"元数据过滤完成，处理了 {len(chunks_for_chroma)} 个文本块用于Chroma。")
        except Exception as e:
            print(f"过滤复杂元数据时发生错误: {e}。将尝试使用原始chunks。")
            chunks_for_chroma = chunks

        print(f"正在对 {len(chunks_for_chroma)} 个文本块进行Embedding并创建Chroma数据库。此过程可能需要较长时间...")
        try:
            _VECTOR_STORE = Chroma.from_documents(
                documents=chunks_for_chroma,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            print(f"新的向量数据库创建成功并已持久化到: {CHROMA_PERSIST_DIR}")
        except Exception as e:
            print(f"创建向量数据库时发生严重错误: {e}")
            _VECTOR_STORE = None  # 确保失败时 _VECTOR_STORE 被设为 None
            return None
    return _VECTOR_STORE


# --- get_qa_chain 函数 (添加 global 声明) ---
def get_qa_chain(force_recreate_vs: bool = False) -> Optional[RetrievalQA]:
    global _QA_CHAIN, _VECTOR_STORE  # <--- 修改这一行, _VECTOR_STORE也可能在递归调用中被修改
    if _QA_CHAIN is not None and not force_recreate_vs:
        print("从内存返回已存在的QA链。")
        return _QA_CHAIN

    # get_vector_store 可能会修改全局的 _VECTOR_STORE
    vector_store = get_vector_store(force_recreate=force_recreate_vs)
    if vector_store is None:  # vector_store 可能是 get_vector_store 返回的局部变量或更新后的全局_VECTOR_STORE
        print("错误: 无法获取向量数据库，因此无法创建QA链。")
        _QA_CHAIN = None  # 确保 _QA_CHAIN 在失败时是 None
        return None

    llm = BaiduErnieLLM()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    try:
        _QA_CHAIN = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        print("QA链创建成功。")
    except Exception as e:
        print(f"创建QA链时发生错误: {e}")
        _QA_CHAIN = None  # 确保 _QA_CHAIN 在失败时是 None
    return _QA_CHAIN


# --- query_rag 函数 (添加 global 声明，如果它打算修改全局变量，但这里主要是读取) ---
def query_rag(question: str) -> Dict[str, Any]:
    global _QA_CHAIN  # <--- 添加这一行 (主要是为了明确是读取全局的)
    print(f"\n接收到查询: {question}")
    if not question or not isinstance(question, str) or not question.strip():
        return {"answer": "请输入一个有效的问题。", "source_documents": []}

    # get_qa_chain 内部会处理 _QA_CHAIN 的状态
    qa_chain_instance = get_qa_chain()  # 获取 qa_chain 实例
    if qa_chain_instance is None:
        return {"answer": "错误: RAG问答链未成功初始化，无法回答问题。", "source_documents": []}

    try:
        print("正在调用QA链进行查询...")
        result = qa_chain_instance.invoke({"query": question})  # 使用获取到的实例
        answer = result.get("result", "未能找到明确的答案。")
        source_docs_raw = result.get("source_documents", [])
        # rag_core.py in query_rag function
        vector_store_instance = get_vector_store()
        if vector_store_instance:
            print(f"--- DEBUG: 直接从vector_store检索与问题 '{question}' 相关的内容 ---")
            try:
                retrieved_docs_direct = vector_store_instance.similarity_search_with_score(question, k=5)
                if retrieved_docs_direct:
                    print(f"--- DEBUG: 直接检索到的前 {len(retrieved_docs_direct)} 个文档 (包含分数): ---")
                    for i, (doc, score) in enumerate(retrieved_docs_direct):
                        print(
                            f"  DOC {i + 1}, Score: {score:.4f}, Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                        print(f"    Content (部分): {doc.page_content[:150]}...\n")
                else:
                    print("--- DEBUG: 直接从vector_store未检索到任何文档。 ---")
            except Exception as e_direct_search:
                print(f"--- DEBUG: 直接从vector_store检索时发生错误: {e_direct_search} ---")
        formatted_sources = []
        if source_docs_raw:
            for doc_item in source_docs_raw:
                if isinstance(doc_item, Document):
                    source_info = {
                        "page_content": doc_item.page_content[:500] + "..." if len(
                            doc_item.page_content) > 500 else doc_item.page_content,
                        "metadata": doc_item.metadata
                    }
                    formatted_sources.append(source_info)
        print(f"答案: {answer}")
        if formatted_sources:
            print(f"参考来源 (部分): {formatted_sources[0]['metadata'] if formatted_sources else '无'}")
        return {"answer": answer, "source_documents": formatted_sources}
    except Exception as e:
        print(f"RAG查询过程中发生错误: {e}")
        return {"answer": f"处理您的问题时发生错误: {e}", "source_documents": []}


# --- initialize_rag_system 函数 (添加 global 声明，如果它打算修改全局变量，但这里主要是读取) ---
def initialize_rag_system(force_recreate_vs: bool = False):
    global _QA_CHAIN  # <--- 添加这一行 (主要是为了明确是读取全局的)
    print("正在初始化RAG系统...")
    try:
        # get_qa_chain 内部会处理 _QA_CHAIN 的状态
        qa_chain_instance = get_qa_chain(force_recreate_vs=force_recreate_vs)  # 获取实例
        if qa_chain_instance:  # 检查获取到的实例
            print("RAG系统初始化成功。")
        else:
            print("警告: RAG系统初始化可能未完全成功，QA链未能创建。")
    except Exception as e:
        print(f"初始化RAG系统失败: {e}")
        raise

def direct_query_llm(query: str) -> Dict:
    """
    直接查询大语言模型，不使用RAG
    
    Args:
        query: 用户查询文本
        
    Returns:
        包含回答的字典
    """
    from baidu_llm import BaiduErnieLLM
    
    llm = BaiduErnieLLM()
    
    # 构造提示词
    prompt = f"""请直接回答以下问题。基于你已有的知识，如果你不确定，请说明。请保持回答简洁、准确。
    
问题: {query}

回答:"""
    
    # 获取回答
    answer = llm.predict(prompt)
    
    return {
        "answer": answer.strip(),
        "source_documents": [],
        "is_direct_answer": True
    }