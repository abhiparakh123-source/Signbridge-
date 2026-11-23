"""Pure CTC greedy decoding helpers, including chunk-boundary state."""
from __future__ import annotations

from operator import index


def _validated_reverse_vocabulary(vocabulary):
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("vocabulary must be a non-empty token-to-ID dictionary")
    reverse = {}
    for token, raw_id in vocabulary.items():
        if not isinstance(token, str) or not token:
            raise ValueError("vocabulary tokens must be non-empty strings")
        if isinstance(raw_id, bool):
            raise ValueError("vocabulary IDs must be non-negative integers")
        try:
            token_id = index(raw_id)
        except TypeError as exc:
            raise ValueError("vocabulary IDs must be non-negative integers") from exc
        if token_id < 0:
            raise ValueError("vocabulary IDs must be non-negative integers")
        if token_id in reverse:
            raise ValueError("vocabulary IDs must be unique")
        reverse[token_id] = token
    if 0 not in reverse:
        raise ValueError("vocabulary must reserve token ID 0 for CTC blank")
    return reverse


def decode_frame_ids_incremental(frame_ids, vocabulary, previous_frame_id=None):
    """Decode newly arrived frame IDs and preserve CTC repeat state.

    Blank ID 0 resets repeat suppression. The previous raw frame ID must be
    carried across chunks so a token spanning a chunk boundary is not duplicated.
    Inputs are required to be integer IDs; silently truncating floats can turn
    corrupt model output into a plausible but incorrect gloss.
    """
    reverse = _validated_reverse_vocabulary(vocabulary)
    if previous_frame_id is not None:
        if isinstance(previous_frame_id, bool):
            raise ValueError("previous frame ID must be an integer in the vocabulary")
        try:
            previous = index(previous_frame_id)
        except TypeError as exc:
            raise ValueError("previous frame ID must be an integer in the vocabulary") from exc
        if previous not in reverse:
            raise ValueError("previous frame ID is absent from vocabulary")
    else:
        previous = None

    emitted = []
    for raw in frame_ids:
        if isinstance(raw, bool):
            raise ValueError("frame IDs must be integers")
        try:
            token_id = index(raw)
        except TypeError as exc:
            raise ValueError("frame IDs must be integers") from exc
        if token_id not in reverse:
            raise ValueError(f"token ID {token_id} is absent from vocabulary")
        if token_id != previous and token_id != 0:
            emitted.append(reverse[token_id])
        previous = token_id
    return emitted, previous
