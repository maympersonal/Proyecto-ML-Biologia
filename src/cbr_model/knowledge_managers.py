from typing import List, Tuple


def knn_manager(classifications: List[Tuple[float, str]], k: int = 20) -> str | None:
    classifications.sort(key=lambda x: x[0])
    classifications = classifications[:k]

    count = {}
    for _, label in classifications:
        if label in count:
            count[label] += 1
        else:
            count[label] = 1

    if max(count.values()) / k > 0.4:
        return max(count, key=count.get)  # type: ignore
    return None
