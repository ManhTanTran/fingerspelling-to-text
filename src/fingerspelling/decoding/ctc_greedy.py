"""Greedy CTC decoding with an explicit vocabulary and blank index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def collapse_repeats_and_blanks(path: Sequence[int], blank_index: int) -> list[int]:
    """Collapse consecutive repeats, then remove blank labels.

    A blank resets the repeat run, so ``A, blank, A`` correctly decodes as
    ``AA`` while ``A, A`` decodes as ``A``.
    """
    collapsed: list[int] = []
    previous: int | None = None
    for raw_index in path:
        index = int(raw_index)
        if index != previous and index != blank_index:
            collapsed.append(index)
        previous = index
    return collapsed


def _token_for(index: int, id_to_token: Sequence[str] | Mapping[int, str]) -> str:
    try:
        token = id_to_token[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"class index {index} is missing from the supplied vocabulary") from exc
    if not isinstance(token, str):
        raise ValueError(f"vocabulary entry {index} must be a string")
    return token


def greedy_decode(
    logits: Any,
    id_to_token: Sequence[str] | Mapping[int, str],
    blank_index: int,
    input_length: int | None = None,
) -> str:
    """Greedily decode one ``[time, classes]`` sequence of logits/log-probs.

    ``id_to_token`` and ``blank_index`` are mandatory to prevent silently
    assuming a token order or CTC blank convention.
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu()
    if hasattr(logits, "tolist"):
        logits = logits.tolist()
    if not isinstance(logits, Sequence) or isinstance(logits, (str, bytes)):
        raise ValueError("logits must be a [time, classes] sequence")
    time_steps = len(logits)
    if input_length is None:
        input_length = time_steps
    if input_length < 0 or input_length > time_steps:
        raise ValueError("input_length must be between zero and the number of time steps")

    path: list[int] = []
    for frame in logits[:input_length]:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)) or not frame:
            raise ValueError("each time step must contain at least one class score")
        path.append(max(range(len(frame)), key=frame.__getitem__))

    token_ids = collapse_repeats_and_blanks(path, blank_index)
    return "".join(_token_for(index, id_to_token) for index in token_ids)

