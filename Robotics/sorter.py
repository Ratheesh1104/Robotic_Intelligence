"""
The orchestrator — ties Jev, LLM, and the robot together.

Flow for each item:
  1. Jev classifies → high confidence? → Jev's decision directly
  2. Jev classifies → low confidence? → LLM reasons → Jev finalizes
  3. Robot executes the decision (pick and place)

Jev is the only thing that ever produces the final bin command.
"""

import logging
import time
from typing import List

from config import SystemConfig
from models.item import Item
from models.decision import Decision, ConfidenceLevel
from classifiers.jev import JevClassifier
from classifiers.llm_reasoner import LLMReasoner
from simulation.mujoco_env import MuJoCoEnv
from simulation.robot_arm import RobotArm

logger = logging.getLogger(__name__)


class Sorter:
    """
    Main sorting orchestrator.
    Coordinates Jev, LLM, and the MuJoCo robot arm.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.jev = JevClassifier(config.jev)
        self.llm = LLMReasoner(config.reasoner)
        self.env = MuJoCoEnv(config.mujoco)
        self.arm = RobotArm(self.env, config.mujoco)
        self.decisions: List[Decision] = []

    def run(self, items: List[Item]):
        """
        Sort all items through the dual-layer pipeline,
        then execute each decision on the robot.
        """
        logger.info(f"Starting sort for {len(items)} items")
        self.env.launch_viewer()

        # Let the sim settle
        for _ in range(500):
            self.env.step()
        self.env.sync_viewer()

        for i, item in enumerate(items):
            logger.info(f"\n{'='*50}\nProcessing item {i+1}/{len(items)}: {item.id}\n{'='*50}")

            # ── Step 1: Jev classifies ─────────────────────────────
            decision = self.jev.classify(item)

            # ── Step 2: If low confidence → LLM reasons → Jev finalizes
            if decision.confidence_level == ConfidenceLevel.LOW:
                print(f"\n  [THINKING] LLM reasoning about {item.id}...")
                thought = self.llm.reason(item)

                # Show thought process on screen (the "think out loud" moment)
                print(f"\n  [THOUGHT PROCESS] {thought}\n")

                # Jev finalizes — converts reasoning to structured decision
                decision = self.jev.finalize(item, thought)
            else:
                # Fast path — instant decision
                print(f"\n  [FAST] {item.id} → Bin {decision.bin_id} ({decision.bin_name})\n")

            self.decisions.append(decision)

            # ── Step 3: Robot executes ─────────────────────────────
            item_body = f"item_{i}"
            print(f"  [ROBOT] Moving {item.id} to Bin {decision.bin_id}...")
            self.arm.pick_and_place(item_body, decision.bin_id)
            print(f"  [DONE] {item.id} placed in Bin {decision.bin_id}\n")

            # Small pause between items for visibility
            time.sleep(0.5)

        # Summary
        self._print_summary(items)
        self.env.close()

    def _print_summary(self, items: List[Item]):
        print(f"\n{'='*50}")
        print("SORTING COMPLETE")
        print(f"{'='*50}")
        for item, dec in zip(items, self.decisions):
            path = "FAST" if not dec.used_llm else "LLM→Jev"
            print(f"  {item.id:10s} → Bin {dec.bin_id} ({dec.bin_name:10s}) via {path:8s} conf={dec.confidence:.2f}")
        print(f"{'='*50}\n")