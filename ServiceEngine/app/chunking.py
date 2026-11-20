import re

from app.config import CHUNK_MAX_CHARS_BART, CHUNK_OVERLAP_SENTENCES

MIN_CHUNK_CHARS = 50  # a trailing chunk shorter than this is merged into the previous one


def chunk_transcript(transcript: str, max_chunk_chars: int = CHUNK_MAX_CHARS_BART,
                     overlap_sentences: int | None = None) -> list[str]:
    """
    Packs whole sentences into chunks of <= max_chunk_chars (overlap included).
    Each chunk after the first starts with the last `overlap_sentences` sentences of the
    previous chunk (default: CHUNK_OVERLAP_SENTENCES). About 20% of the size limit is
    reserved for that overlap, so chunks never exceed the limit.
    """
    k = CHUNK_OVERLAP_SENTENCES if overlap_sentences is None else overlap_sentences
    budget = max_chunk_chars // 5 if k > 0 else 0       # room kept for the overlap
    limit = max_chunk_chars - budget

    sentences = re.split(r"(?<=[.!?\u0964])\s+", transcript.strip())   # \u0964 = Devanagari danda (।)

    # 1) group whole sentences into base chunks (lists of sentences)
    groups, current, size = [], [], 0
    for sent in sentences:
        while len(sent) > limit:                      # no punctuation: hard split
            if current:
                groups.append(current)
                current, size = [], 0
            groups.append([sent[:limit]])
            sent = sent[limit:]
        add = len(sent) + (1 if current else 0)
        if current and size + add > limit:
            groups.append(current)
            current, size = [sent], len(sent)
        else:
            current.append(sent)
            size += add
    if current:
        groups.append(current)

    # 2) a tiny trailing group is merged into the previous one instead of dropped
    if len(groups) > 1 and len(" ".join(groups[-1])) < MIN_CHUNK_CHARS:
        groups[-2].extend(groups.pop())

    # 3) prepend the tail of the previous group as overlap
    chunks = []
    for i, group in enumerate(groups):
        prefix, used = [], 0
        if i > 0 and k > 0:
            for s in reversed(groups[i - 1][-k:]):
                if used + len(s) + 1 > budget:
                    break
                prefix.insert(0, s)
                used += len(s) + 1
        chunks.append(" ".join(prefix + group))
    return chunks