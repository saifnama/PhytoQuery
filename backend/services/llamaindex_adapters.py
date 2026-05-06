"""LlamaIndex adapters for PhytoQuery's existing LLM and embedding wrappers.

Two adapters live here:

* ``PhytoQueryLLM`` — async-first ``CustomLLM`` wrapping our internal
  ``OllamaLLM`` so the Groq → OpenRouter → Ollama provider routing and
  ``RAGProviderAuthError`` / ``RAGLLMTimeoutError`` mapping survive
  unchanged. Sync ``complete`` / ``chat`` intentionally raise — every
  invocation routes through FastAPI handlers, which are already async,
  and we don't want the ``asyncio.run`` from a running loop footgun.

* ``PhytoQueryLIEmbedding`` — ``BaseEmbedding`` wrapping
  ``PhytoQueryEmbeddings`` so the Qwen3-4B → bge-m3 fallback chain, MRL
  truncation, and instruction-aware query prompts continue to work
  inside LlamaIndex's pipeline.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from pydantic import Field, PrivateAttr

from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.embeddings import BaseEmbedding


def _coerce_role(role: Any) -> str:
    """Translate a LlamaIndex ``MessageRole`` (or string) into the OpenAI-
    style role string our ``OllamaLLM`` expects ("system" / "user" /
    "assistant")."""
    if hasattr(role, "value"):
        return str(role.value)
    return str(role)


class PhytoQueryLLM(CustomLLM):
    """Async-first LlamaIndex LLM facade over our ``OllamaLLM``.

    Streaming is unimplemented — none of our call sites stream tokens.
    """

    context_window: int = Field(default=4096, description="Context window in tokens")
    num_output: int = Field(default=2048, description="Max output tokens per call")
    model_name: str = Field(default="phytoquery-llm")
    request_timeout: Optional[float] = Field(
        default=None,
        description="Default per-call timeout in seconds; can be overridden via kwargs.",
    )

    _ollama_llm: Any = PrivateAttr()

    def __init__(
        self,
        ollama_llm: Any,
        context_window: int = 4096,
        num_output: int = 2048,
        request_timeout: Optional[float] = None,
        **data: Any,
    ) -> None:
        super().__init__(
            context_window=context_window,
            num_output=num_output,
            request_timeout=request_timeout,
            model_name=getattr(ollama_llm, "model", "phytoquery-llm"),
            **data,
        )
        self._ollama_llm = ollama_llm

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
        )

    # --- Sync paths: explicitly disabled. ---------------------------------
    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        raise NotImplementedError("PhytoQueryLLM is async-only; use acomplete().")

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        raise NotImplementedError("PhytoQueryLLM does not implement streaming.")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError("PhytoQueryLLM is async-only; use achat().")

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        raise NotImplementedError("PhytoQueryLLM does not implement streaming.")

    # --- Async paths: the only paths actually used. -----------------------
    @llm_completion_callback()
    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        timeout = kwargs.pop("timeout_seconds", self.request_timeout)
        response = await self._ollama_llm.invoke(
            prompt=prompt,
            timeout_seconds=timeout,
        )
        return CompletionResponse(text=response.content.strip())

    async def astream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("PhytoQueryLLM does not implement streaming.")

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        timeout = kwargs.pop("timeout_seconds", self.request_timeout)
        msg_list = [
            {"role": _coerce_role(m.role), "content": m.content or ""}
            for m in messages
        ]
        response = await self._ollama_llm.invoke(
            messages=msg_list,
            timeout_seconds=timeout,
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content.strip(),
            ),
        )

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError("PhytoQueryLLM does not implement streaming.")


class PhytoQueryLIEmbedding(BaseEmbedding):
    """LlamaIndex ``BaseEmbedding`` facade over ``PhytoQueryEmbeddings``.

    Delegates query/document encoding to the wrapped object so the
    Qwen3 → bge-m3 fallback, MRL truncation, and instruction-aware query
    prompts (``prompt_name="query"`` / custom domain instruction) keep
    working inside LlamaIndex.
    """

    _embeddings: Any = PrivateAttr()

    def __init__(
        self,
        embeddings: Any,
        embed_batch_size: int = 16,
        **data: Any,
    ) -> None:
        super().__init__(
            model_name=getattr(embeddings, "model_name", "phytoquery-embedding"),
            embed_batch_size=embed_batch_size,
            **data,
        )
        self._embeddings = embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embeddings.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embeddings.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(list(texts))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)
