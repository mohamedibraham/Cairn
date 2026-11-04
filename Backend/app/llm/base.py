from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel



"""The provider-agnostic contract of the LLM layer (the "LLM Interface").

Nothing in this module knows about any vendor SDK. Application code depends on
"LLMClient" and the plain data types below (Dependency Inversion); concrete
providers live in "llm/providers/" and are created by "llm/factory.py".
"""


Role = Literal["user", "assistant"]
FinishReason = Literal["stop", "length", "safety", "other"]

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class GenerationParams:
    """Per-request overrides. None means "use the configured default"."""
    
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    stop: tuple = ()


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple
    system: str | None = None
    params: GenerationParams = field(default_factory=GenerationParams)

    @classmethod
    def from_prompt(cls, prompt: str, *, system: str | None = None,
                    params: GenerationParams | None = None) -> "LLMRequest":
        return cls(messages=(Message("user", prompt),), system=system,
                   params=params or GenerationParams())


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: FinishReason = "stop"


@dataclass(frozen=True)
class StructuredResponse(Generic[T]):
    parsed: T
    response: LLMResponse


class LLMClient(ABC):
    """Contract every provider (and every decorator) implements."""

    provider: str
    model: str

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        pass

    @abstractmethod
    def generate_structured(self, request: LLMRequest, schema: type) -> StructuredResponse:
        pass

    def complete(self, prompt: str, *, system: str | None = None,
                 params: GenerationParams | None = None) -> LLMResponse:
        
        return self.generate(LLMRequest.from_prompt(prompt, system=system, params=params))
