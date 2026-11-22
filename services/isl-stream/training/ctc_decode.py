"""Pure CTC greedy decoding helpers, including chunk-boundary state."""


def decode_frame_ids_incremental(frame_ids, vocabulary, previous_frame_id=None):
    """Decode newly arrived frame IDs and preserve CTC repeat state.

    Blank ID 0 resets repeat suppression. The previous raw frame ID must be
    carried across chunks so a token spanning a chunk boundary is not duplicated.
    """
    reverse = {index: token for token, index in vocabulary.items()}
    emitted = []
    previous = previous_frame_id
    for raw in frame_ids:
        token_id = int(raw)
        if token_id < 0 or token_id not in reverse:
            if token_id != 0:
                raise ValueError("model emitted a token absent from its vocabulary")
        if token_id != previous and token_id != 0:
            emitted.append(reverse[token_id])
        previous = token_id
    return emitted, previous
