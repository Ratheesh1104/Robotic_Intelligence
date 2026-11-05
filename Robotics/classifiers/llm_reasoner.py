"""
LLM reasoner — the slow path.
Bigger model for chain-of-thought. Returns plain reasoning, NOT a decision.
Jev finalizes everything.
"""

import logging

from openai import OpenAI

from config import ReasonerConfig
from models.item import Item

logger = logging.getLogger(__name__)


class LLMReasoner:
    """Local quantized LLM for reasoning only."""

    def __init__(self, config: ReasonerConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            timeout=config.llm.timeout,
        )
        logger.info(
            f"LLM Reasoner initialized — model={config.llm.model_name}, "
            f"endpoint={config.llm.base_url}"
        )

    def reason(self, item: Item) -> str:
        prompt = self._build_prompt(item)
        try:
            resp = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            thought = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM reasoner call failed: {e}")
            thought = f"Reasoning unavailable (error: {e})."

        logger.info(f"[LLM] Reasoning for {item.id}:\n{thought}")
        return thought

    def _build_prompt(self, item: Item) -> str:
        return (
            f"You are helping a robot sort items into bins.\n"
            f"Bins: A=Heavy(>500g), B=Fragile, C=Standard, D=Recycle\n\n"
            f"An item was flagged as unclear by the fast classifier. "
            f"Think step by step about which bin it belongs in.\n"
            f"Consider weight, category, condition, and any label mismatches.\n\n"
            f"Item: weight={item.weight_grams}g "
            f"category={item.category.value} "
            f"condition={item.condition.value}"
            + (f" label_says={item.label_text}" if item.label_text else "")
            + "\n\nThink out loud, then state your recommended bin."
        )