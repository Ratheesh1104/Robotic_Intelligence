"""
Before running, start llama.cpp servers:

    # Terminal 1 — Jev classification model
    llama-server \
        -m models/Jev-Style-0.8B-Decision-v3-GGUF \
        --port 8001 --host 127.0.0.1 -ngl 99

    # Terminal 2 — Reasoning model (bigger)
    llama-server \
        -m models/mistral-7b-instruct-v0.2-Q5_K_M.gguf \
        --port 8002 --host 127.0.0.1 -ngl 99

Then:
    python main.py
"""

import logging
import sys

from config import SystemConfig, JevConfig, ReasonerConfig, LLMConfig, MuJoCoConfig
from models.item import Item, ItemCategory, ItemCondition
from sorter import Sorter

# Windows consoles default to cp1252, which cannot encode the arrows in the
# output or whatever the LLM happens to emit — and it only bites once output is
# redirected (python main.py > log.txt). Force UTF-8 before anything prints.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)


def main():
    config = SystemConfig(
        jev=JevConfig(
            llm=LLMConfig(
                base_url="http://127.0.0.1:8001/v1",
                model_name="Jev-Style-0.8B-Decision-v3-GGUF",
                temperature=0.0,
                max_tokens=64,
            ),
            confidence_threshold=0.85,
        ),
        reasoner=ReasonerConfig(
            llm=LLMConfig(
                base_url="http://127.0.0.1:8002/v1",
                model_name="Llama-3.2-1B-Instruct-Q4_0.gguf",
                temperature=0.3,
                max_tokens=512,
            ),
        ),
        mujoco=MuJoCoConfig(
            scene_xml="scenes/sorting_scene.xml",
        ),
    )

    items = [
        Item(id="ITM-017", weight_grams=340,
             category=ItemCategory.HARDWARE, condition=ItemCondition.NEW),

        Item(id="ITM-023", weight_grams=120,
             category=ItemCategory.FRAGILE, condition=ItemCondition.NEW),

        Item(id="ITM-055", weight_grams=210,
             category=ItemCategory.HARDWARE, condition=ItemCondition.NEW),

        Item(id="ITM-031", weight_grams=890,
             category=ItemCategory.UNKNOWN, condition=ItemCondition.USED),

        Item(id="ITM-042", weight_grams=45,
             category=ItemCategory.ELECTRONIC,
             condition=ItemCondition.MISMATCH, label_text="fragile"),
    ]

    sorter = Sorter(config)
    sorter.run(items)


if __name__ == "__main__":
    main()