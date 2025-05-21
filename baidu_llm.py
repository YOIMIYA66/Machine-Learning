# baidu_llm.py
import time
from typing import Any, List, Mapping, Optional, Dict

from openai import OpenAI, APIError, APITimeoutError  # 导入 APITimeoutError

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from config import (
    AI_STUDIO_API_KEY,
    AI_STUDIO_BASE_URL,
    BAIDU_LLM_MODEL_NAME,
    BAIDU_EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE
)

if not AI_STUDIO_API_KEY:
    raise ValueError("AI_STUDIO_API_KEY 未找到。请在 .env 文件中或作为环境变量设置。")

# 客户端初始化应该在try-except块内，以便捕获初始化失败
try:
    client = OpenAI(
        api_key=AI_STUDIO_API_KEY,
        base_url=AI_STUDIO_BASE_URL,
        timeout=120.0,  # 增加超时时间到120秒
    )
except Exception as e:
    # 在这里打印错误并重新抛出，或者根据需要处理
    print(f"初始化OpenAI客户端失败: {e}")
    # 可以选择 raise e 或者 sys.exit(1) 如果这是关键组件
    raise  # 重新抛出异常，让调用者知道初始化失败了


class BaiduErnieEmbeddings(Embeddings):
    model_name: str = BAIDU_EMBEDDING_MODEL_NAME
    api_batch_size: int = EMBEDDING_BATCH_SIZE
    # 更新 default_dimension 以匹配 bge-large-zh (1024维)
    # 直接设置为 1024，因为你的配置指定了 BAIDU_EMBEDDING_MODEL_NAME = "bge-large-zh"
    # 除非你有逻辑让 BAIDU_EMBEDDING_MODEL_NAME 动态改变且需要不同的维度
    default_dimension: int = 1024  # 假设 BAIDU_EMBEDDING_MODEL_NAME 固定为 "bge-large-zh"

    def _call_baidu_embedding_api(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.api_batch_size):
            batch_texts = texts[i:i + self.api_batch_size]

            valid_batch_texts = []
            placeholder_indices = []
            # 索引原始批次中的位置
            original_indices_map = {}  # 用于映射有效文本回原始批次的位置
            current_valid_idx = 0

            for original_idx, text_item in enumerate(batch_texts):
                if isinstance(text_item, str) and text_item.strip():
                    # 确保文本长度不超过bge-large-zh的字符限制 (文档建议512 tokens, 约等于 1000-1500 中文字符)
                    # 这里的2000字符截断是一个粗略的保护措施
                    valid_batch_texts.append(text_item[:2000])
                    original_indices_map[current_valid_idx] = original_idx
                    current_valid_idx += 1
                else:
                    print(
                        f"警告: 在Embedding批次中发现无效文本 (空或非字符串): '{text_item}' (在原始批次索引 {original_idx})。将使用占位符。")
                    # 记录原始索引，稍后填充占位符
                    placeholder_indices.append(original_idx)

            batch_embeddings_results = [[0.0] * self.default_dimension] * len(batch_texts)  # 先用占位符初始化整个批次的结果

            if valid_batch_texts:
                try:
                    # print(f"DEBUG: Calling client.embeddings.create with model='{self.model_name}' for {len(valid_batch_texts)} texts.")
                    response = client.embeddings.create(
                        model=self.model_name,
                        input=valid_batch_texts
                    )
                    # print(f"DEBUG: API response received. Data length: {len(response.data)}")
                    # 将有效的 embedding 结果放回它们在原始 batch_texts 中的位置
                    for valid_idx, item in enumerate(response.data):
                        original_batch_idx = original_indices_map.get(valid_idx)
                        if original_batch_idx is not None:
                            batch_embeddings_results[original_batch_idx] = item.embedding
                        else:
                            # 这不应该发生，如果发生了说明 original_indices_map 逻辑有问题
                            print(f"错误: 无法找到有效嵌入结果 {valid_idx} 在原始批次中的映射。")

                except APITimeoutError:  # 已导入
                    print(f"Embedding API 调用超时，批次起始文本: '{valid_batch_texts[0][:50]}...'")
                    # 占位符已经预先填充，无需额外操作
                except APIError as e:
                    print(f"Embedding API 调用错误，批次起始文本 '{valid_batch_texts[0][:50]}...': {e}")
                    print(f"错误详情: status_code={e.status_code}, response={e.response}, body={e.body}")
                    # 占位符已经预先填充
                except Exception as e:
                    print(f"Embedding API 调用时发生未知错误: {e}")
                    # 占位符已经预先填充

            all_embeddings.extend(batch_embeddings_results)
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 确保所有元素都是字符串，并进行截断
        sanitized_texts = [str(text)[:2000] if text is not None and isinstance(text, str) else "" for text in texts]
        return self._call_baidu_embedding_api(sanitized_texts)

    def embed_query(self, text: str) -> List[float]:
        if not text or not isinstance(text, str) or not text.strip():
            print(f"警告: embed_query 收到无效文本: '{text}'。返回零向量。")
            return [0.0] * self.default_dimension
        sanitized_text = str(text)[:2000]
        result = self._call_baidu_embedding_api([sanitized_text])
        return result[0] if result else ([0.0] * self.default_dimension)


class BaiduErnieLLM(LLM):
    model_name: str = BAIDU_LLM_MODEL_NAME
    temperature: float = 0.1
    top_p: Optional[float] = 0.7  # 保持可选
    max_tokens: Optional[int] = 2048  # 保持可选

    @property
    def _llm_type(self) -> str:
        return "baidu_ernie_openai_compatible"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,  # 未使用，但符合接口
            **kwargs: Any,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        # 只有当这些参数不为 None 时才添加到 api_params
        if self.top_p is not None:
            api_params["top_p"] = self.top_p
        if self.max_tokens is not None:
            api_params["max_tokens"] = self.max_tokens
        if stop:  # stop 也是可选的
            api_params["stop"] = stop

        api_params.update(kwargs)  # 合并额外参数
        # 移除值为 None 的参数，因为 OpenAI API 可能不喜欢 None 值
        api_params = {k: v for k, v in api_params.items() if v is not None}

        max_retries = 2
        retry_count = 0
        last_exception = None  # 用于存储最后一次异常

        while retry_count <= max_retries:
            try:
                # print(f"DEBUG: Calling LLM with params: {api_params}")
                completion = client.chat.completions.create(
                    messages=messages,
                    **api_params
                )
                content = completion.choices[0].message.content
                return content if content is not None else ""  # 确保返回字符串
            except APITimeoutError as e_timeout:  # 捕获具体的超时错误
                last_exception = e_timeout
                retry_count += 1
                print(f"调用百度LLM API超时 (尝试 {retry_count}/{max_retries + 1})，Prompt起始: '{prompt[:50]}...'")
                if retry_count <= max_retries:
                    print(f"正在重试...")
                    time.sleep(1)  # 添加少量延迟再重试
                    continue
                # 如果所有重试都失败，则在循环外抛出或返回错误
                print("LLM API 请求多次尝试后仍然超时。")
                # return f"错误：LLM API 请求多次尝试后仍然超时 - {last_exception}" # 或者抛出
                raise TimeoutError(f"LLM API 请求多次尝试后仍然超时 - {last_exception}") from last_exception
            except APIError as e_api:  # 捕获具体的API错误
                last_exception = e_api
                retry_count += 1
                error_message = f"调用百度LLM API时发生错误: {e_api.message if hasattr(e_api, 'message') else str(e_api)}"
                print(error_message)
                baidu_error_msg = ""  # 初始化
                if e_api.body and isinstance(e_api.body, dict):
                    baidu_error_msg = e_api.body.get("error_msg") or e_api.body.get("error", {}).get("message")
                    if baidu_error_msg:
                        error_detail = f"百度LLM API错误: {baidu_error_msg} (错误码: {e_api.body.get('error_code')})"
                        print(f"详细错误: {error_detail}")
                        error_message = error_detail  # 使用更具体的百度错误信息

                print(f"请求体: messages={messages}, params={api_params}")

                # 对于某些可重试的错误类型 (例如，速率限制通常有特定错误码或消息)
                # 你原来的 "rate limit" in str(e).lower() 可能不够精确
                # 这里简化为只有超时才重试，其他APIError直接失败或根据错误码判断是否重试
                if retry_count <= max_retries and (
                        "rate_limit_exceeded" in str(e_api).lower() or e_api.status_code == 429):  # 示例: 检查速率限制
                    print(f"遇到可重试的API错误，正在重试 (尝试 {retry_count}/{max_retries + 1})...")
                    time.sleep(retry_count * 2)  # 增加退避时间
                    continue
                else:
                    print(f"API错误不可重试或已达最大重试次数: {error_message}")
                    # return error_message # 或者抛出
                    raise e_api from last_exception  # 重新抛出APIError
            # 修正点：下面的 except Exception as e 应该与 try 对齐
            except Exception as e_unknown:  # 捕获其他所有未知异常
                last_exception = e_unknown
                print(f"调用百度LLM API时发生未知错误 (尝试 {retry_count + 1}/{max_retries + 1}): {e_unknown}")
                # 未知错误通常不建议立即重试，除非有特定原因
                # 如果决定不重试未知错误：
                # return f"错误：LLM 调用时发生未知错误 - {e_unknown}"
                # 或者，如果也想重试未知错误：
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"正在重试未知错误...")
                    time.sleep(1)
                    continue
                print("LLM API 调用时发生未知错误，已达最大重试次数。")
                # return f"错误：LLM 调用时发生未知错误，已达最大重试次数 - {last_exception}"
                raise RuntimeError(
                    f"LLM API 调用时发生未知错误，已达最大重试次数 - {last_exception}") from last_exception

        # 如果循环结束仍未成功返回（理论上应该在循环内返回或抛出异常）
        # 这段代码实际上不应该被执行到，因为上面的逻辑会 return 或 raise
        return f"错误：LLM 调用在所有重试后均失败。最后一次错误: {last_exception}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }