"""Heuristic attribution models: First-Touch, Last-Touch, and Linear."""

from typing import Any, Dict, List


def compute_heuristic_attribution(
    paths: List[Dict[str, Any]],
    channels: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute first-touch, last-touch, and linear attribution shares across conversion paths."""
    first_touch_counts: Dict[str, float] = {c: 0.0 for c in channels}
    last_touch_counts: Dict[str, float] = {c: 0.0 for c in channels}
    linear_counts: Dict[str, float] = {c: 0.0 for c in channels}

    total_conversions = 0.0

    for path in paths:
        if not path.get("converted", False):
            continue
        route = path["path"]  # list of channel names
        weight = path.get("weight", 1.0)
        if not route:
            continue

        total_conversions += weight

        # First Touch
        first_touch_counts[route[0]] += weight

        # Last Touch
        last_touch_counts[route[-1]] += weight

        # Linear
        n_steps = len(route)
        for ch in route:
            linear_counts[ch] += weight / n_steps

    def normalize(counts: Dict[str, float]) -> Dict[str, float]:
        total = sum(counts.values())
        if total == 0:
            return {c: 1.0 / len(channels) for c in channels}
        return {c: round(v / total, 4) for c, v in counts.items()}

    return {
        "first_touch": normalize(first_touch_counts),
        "last_touch": normalize(last_touch_counts),
        "linear": normalize(linear_counts),
    }
