"""
Jev classifier — uses Jev-Style-0.8B-Decision-v3-GGUF.

Jev is a text classification model, not a chatbot. Its label vocabulary is
" yes" / " no": given a question about an item it answers with one token. It
cannot emit JSON, prose, or a bin letter, so it is driven the way a classifier
is meant to be driven — one binary probe per bin, asked in priority order, and
the first `yes` decides the bin.

Confidence is not something Jev is asked to write down; it is read off the
model's own token distribution:  p(yes) / (p(yes) + p(no)).

Two jobs:
  1. classify()   → probe the item directly, output bin + confidence
  2. finalize()   → probe the LLM's reasoning, output final bin + confidence

Every bin comes from Jev. There is no Python fallback and no default bin: if
Jev commits to nothing, that is reported as low confidence (which routes the
item to the LLM) or raised as JevError.
"""

import json
import logging
import math
import urllib.error
import urllib.request

from config import JevConfig
from models.item import Item
from models.decision import Decision, ConfidenceLevel

logger = logging.getLogger(__name__)


# ─── sorting policy ──────────────────────────────────────────────────
# The questions ARE the policy, and they are all Jev ever gets asked. Probing
# order is the tie-break order: the first bin Jev says "yes" to wins. Edit the
# wording here to change how items are sorted — no branching logic to touch.
#
# {item} is replaced with the item's own data.
BIN_PROBES = [
    ("A", "Does this item weigh over 500 grams?"),
    ("B", "Is this item fragile, or does its label claim it is fragile?"),
    ("D", "Should this item be scrapped for recycling?"),
]

# The residual class. Standard is not a probe because "standard" is defined as
# none of the above — asking Jev "is this ordinary?" scores ~0.4 on everything
# and the bin becomes unreachable. Reaching it still takes a `no` from Jev on
# every probe above; confidence is how firmly it ruled them all out.
RESIDUAL_BIN = "C"

FINALIZE_PROBE = "Does this reasoning conclude that the item belongs in the {bin_name} bin?"


class JevError(RuntimeError):
    """Jev produced nothing usable — no bin was decided."""


class JevClassifier:
    """
    Jev: fast text classification model (0.8B GGUF).
    Runs on its own llama.cpp server instance.
    """

    def __init__(self, config: JevConfig):
        self.config = config
        # Jev is driven through llama.cpp's native /completion, because the
        # OpenAI chat route hides the token probabilities the confidence needs.
        self.endpoint = config.llm.base_url.rstrip("/")
        if self.endpoint.endswith("/v1"):
            self.endpoint = self.endpoint[: -len("/v1")]
        self.endpoint += "/completion"

        logger.info(
            f"Jev initialized — model={config.llm.model_name}, "
            f"endpoint={self.endpoint}, "
            f"threshold={config.confidence_threshold}"
        )

    # ─── Step 1: classify ────────────────────────────────────────────

    def classify(self, item: Item) -> Decision:
        """
        Classify an item. Returns either:
          - Confident Decision (fast path), or
          - Low-confidence Decision (signals: send to LLM).
        """
        facts = self._item_facts(item)
        bin_id, confidence, verdicts = self._probe_cascade(
            lambda q: f"{facts}\nQuestion: {q}\nAnswer:"
        )
        logger.debug("Jev probes for %s: %s", item.id, verdicts)

        decision = Decision(
            bin_id=bin_id,
            bin_name=self.config.bins[bin_id],
            confidence=confidence,
            confidence_level=(
                ConfidenceLevel.HIGH
                if confidence >= self.config.confidence_threshold
                else ConfidenceLevel.LOW
            ),
        )

        if decision.confidence_level == ConfidenceLevel.HIGH:
            logger.info(
                f"[Jev FAST] {item.id} → Bin {decision.bin_id} "
                f"({decision.confidence:.2f})"
            )
        else:
            logger.info(
                f"[Jev → LLM] {item.id} confidence={decision.confidence:.2f}, "
                f"needs reasoning"
            )
        return decision

    # ─── Step 2: finalize ────────────────────────────────────────────

    def finalize(self, item: Item, thought_process: str) -> Decision:
        """
        Take the LLM's plain-language reasoning and convert it
        into a structured Decision. Jev always does this —
        the LLM never talks to the robot directly.
        """
        facts = self._item_facts(item)
        reasoning = " ".join(thought_process.split())

        def prompt_for(bin_id):
            question = FINALIZE_PROBE.format(bin_name=self.config.bins[bin_id])
            return (
                f"{facts}\nAn assistant reasoned about this item: {reasoning}\n"
                f"Question: {question}\nAnswer:"
            )

        bin_id, confidence, verdicts = self._probe_argmax(prompt_for)
        logger.debug("Jev finalize probes for %s: %s", item.id, verdicts)

        logger.info(
            f"[Jev FINAL] {item.id} → Bin {bin_id} ({confidence:.2f}) via LLM reasoning"
        )
        return Decision(
            bin_id=bin_id,
            bin_name=self.config.bins[bin_id],
            confidence=confidence,
            confidence_level=ConfidenceLevel.FINAL,
            reasoning=f"Jev confirmed {self.config.bins[bin_id]} "
                      f"from the reasoning at p={confidence:.2f}",
            thought_process=thought_process,
        )

    # ─── probing ─────────────────────────────────────────────────────

    def _probe_cascade(self, build_prompt):
        """
        Ask Jev one yes/no question per special bin, in BIN_PROBES order.
        First yes wins. All no means the residual bin, and how firmly Jev
        ruled the others out becomes the confidence.
        """
        verdicts = []
        for bin_id, question in BIN_PROBES:
            if bin_id not in self.config.bins:
                continue
            said_yes, p_yes = self._ask(build_prompt(question))
            verdicts.append((bin_id, round(p_yes, 3)))
            logger.debug("Jev probe %s: yes=%s p=%.3f", bin_id, said_yes, p_yes)
            if said_yes:
                return bin_id, p_yes, verdicts

        if RESIDUAL_BIN not in self.config.bins:
            raise JevError(
                f"Jev said no to every bin and {RESIDUAL_BIN!r} is not a "
                f"configured bin: {verdicts}"
            )
        strongest_rival = max((p for _, p in verdicts), default=0.0)
        return RESIDUAL_BIN, 1.0 - strongest_rival, verdicts

    def _probe_argmax(self, build_prompt):
        """
        Score every bin and let the highest win. Used for finalize, where the
        LLM has already named a bin and Jev only has to recognise which one.
        """
        scored = []
        for bin_id in self.config.bins:
            _, p_yes = self._ask(build_prompt(bin_id))
            scored.append((bin_id, p_yes))
        logger.debug("Jev finalize scores: %s", [(b, round(p, 3)) for b, p in scored])
        bin_id, p_yes = max(scored, key=lambda pair: pair[1])
        return bin_id, p_yes, [(b, round(p, 3)) for b, p in scored]

    @staticmethod
    def _item_facts(item: Item) -> str:
        return (
            f"Item: weight={item.weight_grams}g "
            f"category={item.category.value} "
            f"condition={item.condition.value}"
            + (f" label={item.label_text}" if item.label_text else "")
        )

    def _ask(self, prompt: str):
        """
        One yes/no probe. Returns (answered_yes, p_yes) where p_yes is the
        model's own probability mass on yes, normalised against no.
        """
        payload = {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": self.config.llm.temperature,
            "n_probs": 12,
        }
        request = urllib.request.Request(
            self.endpoint,
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.llm.timeout
            ) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            logger.error(f"Jev call failed: {e}")
            raise JevError(f"Jev unreachable at {self.endpoint}: {e}") from e

        probabilities = body.get("completion_probabilities") or []
        if not probabilities:
            raise JevError(
                f"Jev returned no token probabilities — cannot score this item. "
                f"Is {self.endpoint} serving a completion model? Got: {body.get('content')!r}"
            )

        p_yes, p_no = 0.0, 0.0
        for candidate in probabilities[0].get("top_logprobs") or []:
            token = (candidate.get("token") or "").strip().lower()
            weight = math.exp(candidate.get("logprob", -math.inf))
            if token.startswith("yes"):
                p_yes += weight
            elif token.startswith("no"):
                p_no += weight

        total = p_yes + p_no
        if total <= 0.0:
            raise JevError(
                f"Jev put no probability on yes or no — it is not answering the "
                f"question. Raw answer: {body.get('content')!r}"
            )

        p_yes /= total
        return p_yes > 0.5, p_yes


if __name__ == "__main__":
    # Integration self-check — needs the Jev llama-server up.
    #   python -m classifiers.jev
    from models.item import ItemCategory, ItemCondition

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    jev = JevClassifier(JevConfig())

    # Jev must separate these two on a single probe, or it is not reading input.
    heavy = jev._ask(
        "Item: weight=890g category=unknown condition=used\n"
        "Question: Is this item heavier than 500g?\nAnswer:"
    )
    light = jev._ask(
        "Item: weight=45g category=electronic condition=new\n"
        "Question: Is this item heavier than 500g?\nAnswer:"
    )
    print(f"  890g heavier than 500g? {heavy}")
    print(f"   45g heavier than 500g? {light}")
    assert heavy[0] and not light[0], f"Jev cannot tell 890g from 45g: {heavy} {light}"

    probes = [
        Item("ITM-017", 340, ItemCategory.HARDWARE, ItemCondition.NEW),
        Item("ITM-023", 120, ItemCategory.FRAGILE, ItemCondition.NEW),
        Item("ITM-031", 890, ItemCategory.UNKNOWN, ItemCondition.USED),
        Item("ITM-042", 45, ItemCategory.ELECTRONIC, ItemCondition.MISMATCH, "fragile"),
    ]
    chosen = []
    for probe in probes:
        d = jev.classify(probe)
        print(f"  {probe.summary()}\n    -> Bin {d.bin_id} ({d.bin_name}) conf={d.confidence:.2f}")
        chosen.append(d.bin_id)

    assert len(set(chosen)) > 1, f"Jev put every item in the same bin: {chosen}"
    print("\nJev is reading its input and deciding.")
