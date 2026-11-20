"""
Semantic role labelling for the refinement stage: who did what, to whom, when and where.

Each result is one "frame" (a dict):
    {"predicate": "plant", "agent": "John", "patient": "a tree",
     "time": "yesterday", "location": "in Paris", "negated": False}

Backends (SRL_BACKEND in .env):
    spacy  dependency-parse rules on the English spaCy model: fast, deterministic, English only.
           An approximation of SRL (agent ~ ARG0, patient ~ ARG1, time ~ ARGM-TMP, place ~ ARGM-LOC).
    llm    asks the configured LLM for the same frames; works for any language, can vary between runs.
    off    no roles.
SRL never fails a job: any error gives an empty list.
"""
import json

from app.config import SRL_BACKEND

MAX_FRAMES = 30
_LOC_PREPS = {"in", "at", "on", "near", "inside", "outside", "across", "through", "from", "to", "into", "around"}
_PLACE_PREPS = {"in", "at", "near", "inside", "outside"}
_LOC_ENTS = {"GPE", "LOC", "FAC"}
_TIME_ENTS = {"DATE", "TIME"}


# ---------------------------------------------------------------- spaCy backend
def _span_text(tok) -> str:
    """Noun phrase around a head token: its left modifiers plus any 'of ...' complement."""
    left, right = tok.left_edge.i, tok.i
    while left < right and tok.doc[left].dep_ in {"mark", "cc", "punct"}:   # drop a leading "that", "and"...
        left += 1
    for child in tok.children:
        if child.dep_ == "prep" and child.lower_ == "of":
            right = max(right, child.right_edge.i)
    return tok.doc[left:right + 1].text


def _first(children, deps):
    return next((c for c in children if c.dep_ in deps), None)


def _subject_of(verb):
    """The verb's own subject, or the one it shares with the verb it is coordinated with / complements."""
    subj = _first(verb.children, {"nsubj", "csubj"})
    if subj is None and verb.dep_ in {"conj", "xcomp", "advcl"} and verb.head is not verb:
        subj = _first(verb.head.children, {"nsubj", "csubj"})
    return subj


def _frames_for_sentence(sent) -> list[dict]:
    frames = []
    for verb in sent:
        if verb.pos_ != "VERB":                       # auxiliaries (is, has, will...) are not predicates
            continue
        agent = _subject_of(verb)
        patient = _first(verb.children, {"dobj", "nsubjpass"})
        if patient is not None and patient.dep_ == "nsubjpass":      # passive: "The door was opened by Mary"
            by = _first(verb.children, {"agent"})
            pobj = _first(by.children, {"pobj"}) if by is not None else None
            agent = pobj                                              # (None if there is no "by ...")
        if agent is None and patient is None and _first(verb.children, {"ccomp"}) is None:
            continue

        # a clause as object ("found THAT sleep clears waste", "explains HOW lightning works")
        if patient is None:
            clause = _first(verb.children, {"ccomp"})
            if clause is not None:
                words = [t for t in clause.subtree]
                while words and words[0].dep_ in {"mark", "advmod"} and words[0].i < clause.i:
                    words = words[1:]
                patient_text = " ".join(t.text for t in words)
            else:
                patient_text = None
        else:
            patient_text = _span_text(patient)

        # time / place: prepositions hang on the verb or, often, on its object noun ("a product IN Bangalore")
        holders = list(verb.children) + (list(patient.children) if patient is not None else [])
        time = location = None
        for child in holders:
            if child.dep_ in {"npadvmod", "advmod", "tmod"} and child.ent_type_ in _TIME_ENTS:
                time = time or " ".join(t.text for t in child.subtree)
            elif child.dep_ == "prep" and child.lower_ != "of":     # "of ..." belongs to the noun phrase
                pobj = _first(child.children, {"pobj"})
                if pobj is None:
                    continue
                if pobj.ent_type_ in _TIME_ENTS or any(t.ent_type_ in _TIME_ENTS for t in pobj.subtree):
                    time = time or " ".join(t.text for t in child.subtree)
                elif child.lower_ in _LOC_PREPS and (
                    pobj.ent_type_ in _LOC_ENTS
                    # the small model sometimes mislabels a place (e.g. as PERSON): accept "in/at/near + proper noun"
                    or (pobj.pos_ == "PROPN" and child.lower_ in _PLACE_PREPS)
                ):
                    location = location or " ".join(t.text for t in child.subtree)

        frames.append({
            "predicate": verb.lemma_.lower(),
            "agent": _span_text(agent) if agent is not None else None,
            "patient": patient_text,
            "time": time,
            "location": location,
            "negated": any(c.dep_ == "neg" for c in verb.children),
        })
    return frames


def _spacy_roles(text: str, nlp) -> list[dict]:
    frames = []
    for sent in nlp(text).sents:
        frames.extend(_frames_for_sentence(sent))
    return frames


# ---------------------------------------------------------------- LLM backend
_LLM_SYSTEM = (
    "You are a semantic role labeller. You output only valid JSON, with no commentary."
)


def _llm_roles(text: str) -> list[dict]:
    from app.llm_summarizer import _call_llm        # same provider layer, timeouts and retries

    prompt = (
        "Label the semantic roles of the main events in this text. Return a JSON array; each item has "
        'the keys "predicate" (the verb, base form), "agent" (who did it), "patient" (to whom or what), '
        '"time", "location" and "negated" (true/false). Use null where a role is absent. Copy role '
        f"phrases from the text, in its own language. At most {MAX_FRAMES} items.\n\nTEXT:\n{text}"
    )
    raw = _call_llm(_LLM_SYSTEM, prompt)
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in the model's reply")
    items = json.loads(raw[start:end + 1])
    keys = ("predicate", "agent", "patient", "time", "location")
    frames = []
    for it in items:
        if isinstance(it, dict) and it.get("predicate"):
            frames.append({**{k: (str(it[k]).strip() if it.get(k) else None) for k in keys},
                           "negated": bool(it.get("negated"))})
    return frames


# ---------------------------------------------------------------- public API
def extract_roles(text: str, language: str = "en", nlp=None, backend: str | None = None) -> list[dict]:
    backend = (backend or SRL_BACKEND).lower()
    if backend == "off" or not text.strip():
        return []
    if backend == "spacy":
        if language != "en" or nlp is None:
            return []                                  # the spaCy rules are English-only
        frames = _spacy_roles(text, nlp)
    elif backend == "llm":
        frames = _llm_roles(text)
    else:
        raise ValueError(f"Unknown SRL_BACKEND '{backend}' (use 'spacy', 'llm' or 'off').")

    seen, unique = set(), []                           # drop exact duplicates, keep order
    for f in frames:
        key = tuple(f.values())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[:MAX_FRAMES]


def safe_extract_roles(text: str, language: str = "en", nlp=None, backend: str | None = None) -> list[dict]:
    """Like extract_roles, but never raises: SRL is an extra and must not fail a job."""
    try:
        return extract_roles(text, language, nlp, backend)
    except Exception:
        return []