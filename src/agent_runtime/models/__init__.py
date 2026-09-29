"""Provider-neutral message and response records."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Message:
    role: str
    content: str

@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
