#!/usr/bin/env python3
"""Shared split utilities for run-level train/val/test partitions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split_70_10_20(
    items: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70% / 10% / 20% train / val / test."""
    items_tv, items_test, labels_tv, labels_test = train_test_split(
        items,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )
    val_fraction_of_tv = 0.1 / 0.8
    items_train, items_val, labels_train, labels_val = train_test_split(
        items_tv,
        labels_tv,
        test_size=val_fraction_of_tv,
        stratify=labels_tv,
        random_state=random_state,
    )
    return items_train, items_val, items_test, labels_train, labels_val, labels_test
