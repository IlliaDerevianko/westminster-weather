# ruff: noqa: E501  (regex lexicon: one pattern per line is easier to review)
"""Trigger phrases: how the UK government signals it will (or won't) legislate.

Firmness levels used everywhere in the pipeline:
  2 = firm    "we will legislate", "a bill will be introduced this session"
  1 = medium  "we intend to legislate", "at the earliest opportunity", "draft bill"
  0 = vague   "when parliamentary time allows", "in due course", "keep under review"
Phrases that look like commitments but are NOT (negations, questions, opposition demands)
are listed in EXCLUDE and checked first.
"""

import re

# The subject must be the government speaking ("we", "the government", "ministers").
GOV = r"(?:we|the government|this government|ministers|my department|the department)"
LEG = (
    r"(?:legislat\w*|(?:a|the|new|primary|emergency)?\s*(?:bill|legislation|legislative changes?))"
)

FIRM = [
    rf"\b{GOV}\s+(?:will|shall|are going to)\s+(?:now\s+)?legislate\b",
    rf"\b{GOV}\s+(?:will|shall|are going to)\s+(?:introduce|bring forward|bring in|lay|publish)\s+(?:\w+\s+){{0,3}}?{LEG}",
    rf"\b{GOV}\s+(?:are|is)\s+(?:introducing|bringing forward|legislating)\b",
    r"\blegislation will be (?:introduced|brought forward|laid)\b",
    r"\b(?:a|the) bill will be (?:introduced|brought forward|published)\b",
    r"\bwill be (?:included|set out) in the (?:king|queen)['’]s speech\b",
    r"\b(?:king|queen)['’]s speech (?:will|includes?|included) (?:a|the) \w+ bill\b",
]

MEDIUM = [
    rf"\b{GOV}\s+(?:intend|plan|propose|expect|aim)s?\s+to\s+(?:legislate|introduce|bring forward)\b",
    rf"\b{GOV}\s+(?:are|is|remain)\s+committed to\s+(?:legislating|introducing legislation|bringing forward legislation)\b",
    r"\bat the earliest (?:opportunity|possible opportunity)\b",
    r"\bas soon as parliamentary time (?:allows|permits)\b",
    rf"\b{GOV}\s+will\s+publish\s+(?:a\s+)?draft\s+(?:bill|legislation)\b",
    r"\bdraft (?:bill|legislation) will be published\b",
]

VAGUE = [
    r"\bwhen parliamentary time (?:allows|permits)\b",
    r"\bwhen (?:legislative|parliamentary) time is available\b",
    r"\b(?:in due course|at an appropriate time|in the fullness of time)\b",
    rf"\b{GOV}\s+(?:will|are|is)\s+(?:consider|considering|exploring|looking at)\s+(?:\w+\s+){{0,3}}?(?:legislation|legislative options|a bill)\b",
    rf"\b{GOV}\s+will\s+keep\s+(?:the need for (?:new\s+)?legislation|the case for legislation|this)[^.]{{0,150}}?\bunder review\b",
    r"\bif necessary,?\s+(?:we|the government)\s+will\s+legislate\b",
    rf"\b{GOV}\s+(?:will|are going to)\s+consult\s+on\s+(?:\w+\s+){{0,3}}?(?:legislation|proposals|a bill)\b",
    r"\bwe do not rule out legislation\b",
]

# Checked first: if any match the sentence, it is not a government commitment.
EXCLUDE = [
    r"\bno (?:current )?plans to legislate\b",
    r"\b(?:will not|won['’]t|do not intend to|does not intend to|cannot)\s+legislate\b",
    r"\b(?:will|would|does|did|can|could)\s+(?:the|my right hon\.?|my hon\.?)\s+(?:minister|secretary|government|friend|member)\b",  # questions to ministers
    r"\b(?:why|when|how)\s+(?:will|won['’]t|has|hasn['’]t|is)\s+the government\b",
    r"\b(?:i|we) (?:urge|call on|ask|implore) the government\b",
    r"\bthe government (?:should|must|ought to) (?:legislate|introduce|bring forward)\b",
    r"\b(?:the opposition|the previous government|labour|the conservatives|the liberal democrats) (?:would|will|promised to)\b",
]

# Timing words that make a commitment firmer.
TIMING_FIRM = [
    r"\bthis (?:session|year|autumn|spring|summer|winter)\b",
    r"\bin the (?:coming|next few) (?:weeks|months)\b",
    r"\b(?:early|later) (?:this|next) year\b",
    r"\bby (?:the end of )?(?:january|february|march|april|may|june|july|august|september|october|november|december|20\d\d)\b",
    r"\bin the (?:next|forthcoming) (?:king|queen)['’]s speech\b",
]


def _c(patterns):
    return [re.compile(p, re.I) for p in patterns]


FIRM_RE, MEDIUM_RE, VAGUE_RE, EXCLUDE_RE, TIMING_RE = map(
    _c, (FIRM, MEDIUM, VAGUE, EXCLUDE, TIMING_FIRM)
)
ANY_TRIGGER_RE = FIRM_RE + MEDIUM_RE + VAGUE_RE

# Short phrases for server-side search (Hansard search is keyword-based, so keep them simple).
SEARCH_PHRASES = [
    "we will legislate",
    "we will introduce legislation",
    "we will bring forward legislation",
    "we will introduce a bill",
    "legislation will be introduced",
    "we intend to legislate",
    "when parliamentary time allows",
    "at the earliest opportunity",
    "draft bill",
    "we will bring forward a bill",
    "committed to legislating",
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"'])")


def classify_sentence(sentence):
    """Return (firmness 0/1/2, matched phrase) or (None, None) if not a commitment."""
    s = sentence.strip()
    if s.endswith("?") or any(r.search(s) for r in EXCLUDE_RE):
        return None, None
    draft = re.search(r"\bdraft (?:bill|legislation)\b", s, re.I)
    if draft and not any(v.search(s) for v in VAGUE_RE[:3]):  # a draft is a step short of a bill
        return 1, draft.group(0)
    for level, regs in ((2, FIRM_RE), (1, MEDIUM_RE), (0, VAGUE_RE)):
        for r in regs:
            m = r.search(s)
            if m:
                # "we will legislate ... when parliamentary time allows" is vague, not firm
                if level > 0 and any(v.search(s) for v in VAGUE_RE[:3]):
                    level = 0
                if level == 1 and any(t.search(s) for t in TIMING_RE):
                    level = 2
                return level, m.group(0)
    return None, None


def find_triggers(text, context=1):
    """All commitment sentences in a text, with one sentence of context either side."""
    sents = SENT_SPLIT.split(re.sub(r"\s+", " ", text or ""))
    hits = []
    for k, s in enumerate(sents):
        level, phrase = classify_sentence(s)
        if level is not None:
            snippet = " ".join(sents[max(0, k - context) : k + context + 1])
            hits.append(
                {"sentence": s, "snippet": snippet, "rule_firmness": level, "phrase": phrase}
            )
    return hits
