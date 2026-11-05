from enum import Enum
from dataclasses import dataclass


class ItemCategory(Enum):
    HARDWARE   = "hardware"
    FRAGILE    = "fragile"
    CONSUMABLE = "consumable"
    ELECTRONIC = "electronic"
    UNKNOWN    = "unknown"


class ItemCondition(Enum):
    NEW      = "new"
    USED     = "used"
    DAMAGED  = "damaged"
    MISMATCH = "mismatch"   # mislabeled


@dataclass
class Item:
    """
    One item on the tray.
    All data is attached directly — no camera / vision needed.
    """
    id: str
    weight_grams: float
    category: ItemCategory
    condition: ItemCondition
    label_text: str = ""          # what the item *says* it is (can be wrong)

    @property
    def is_ambiguous(self) -> bool:
        """Quick check — is anything off about this item?"""
        return (
            self.category == ItemCategory.UNKNOWN
            or self.condition in (ItemCondition.DAMAGED, ItemCondition.MISMATCH)
            or (self.label_text and self._label_mismatches())
        )

    def _label_mismatches(self) -> bool:
        """Label says one category, data says another."""
        return self.label_text.lower() != self.category.value.lower()

    def summary(self) -> str:
        return (
            f"Item {self.id}: weight={self.weight_grams}g, "
            f"category={self.category.value}, condition={self.condition.value}"
            + (f", label='{self.label_text}'" if self.label_text else "")
        )