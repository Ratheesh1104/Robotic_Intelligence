from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ConfidenceLevel(Enum):
    HIGH   = "high"     # fast path
    LOW    = "low"      # needs LLM
    FINAL  = "final"    # after Jev finalizes LLM output


@dataclass
class Decision:
    """
    The structured output Jev always produces.
    This is the only thing the robot ever receives.
    """
    bin_id: str                       # A, B, C, D
    bin_name: str                     # Heavy, Fragile, Standard, Recycle
    confidence: float                 # 0.0 – 1.0
    confidence_level: ConfidenceLevel
    reasoning: Optional[str] = None   # filled when LLM was used
    thought_process: Optional[str] = None  # LLM's visible "thinking out loud"

    @property
    def used_llm(self) -> bool:
        return self.thought_process is not None