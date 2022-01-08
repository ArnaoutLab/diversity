"""Tests for metacommunity.diversity."""
from pytest import mark

from metacommunity.diversity import Abundance


if ordered_items is None:
    ordered_items_, item_positions = unique(items, return_inverse=True)
    if len(ordered_items_) != len(ordered_items):
        raise Exception
else:
    ordered_items_ = ordered_items
    item_to_position = {item: pos for pos, item in enumerate(ordered_items_)}
    item_positions = array([item_to_position[item] for item in items])
return (ordered_items_, item_positions)
