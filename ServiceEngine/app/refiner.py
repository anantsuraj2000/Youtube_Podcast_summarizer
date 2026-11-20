import spacy
from app.srl import safe_extract_roles
_nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> list[dict]:
    doc = _nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities


def refine_summary(summary: str) -> dict:
    entities = extract_entities(summary)
    return {
        "summary": summary,
        "entities": entities,
        "entity_count": len(entities),
        "roles": safe_extract_roles(summary, "en", _nlp),
    }