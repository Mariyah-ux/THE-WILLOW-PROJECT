"""Conservative English context gate, not a risk or intent classifier.

Rules inspect original text, before model preprocessing loses punctuation.
A covered clause anywhere wins over binary intent elsewhere (mixed statements).
No raw text or model diagnostics are logged or persisted by this module.
"""
import re
import unicodedata

REVIEW = 'context_review_required'
_ACTION = r'(?:die|kill myself|hurt myself|harm myself|end my life|commit suicide|take my own life)'
_DENIAL = re.compile(r'\bi (?:do not (?:want|intend|plan) to '+_ACTION+r'|(?:will|would|could) not (?:kill|hurt|harm) myself|am not (?:suicidal|thinking (?:about|of) suicide|planning (?:suicide|to '+_ACTION+r'))|have no (?:intention|intent|plans?) (?:of (?:suicide|hurting myself|killing myself)|to '+_ACTION+r'))\b')
_LIVE = re.compile(r'\bi (?:want|wish|intend|plan|choose) to (?:live|stay alive|keep living)\b(?!\s+(?:in|at|on|with|near|by|there|here|abroad|alone|forever|like)\b)')
_TOPIC = re.compile(r'\b(?:suicide|suicidal|kill (?:myself|yourself|himself|herself)|end my life|hurt myself|harm myself|want to die)\b')
_QUOTE = re.compile(r'"([^"\n]{1,1000})"|(?<!\w)\x27([^\x27\n]{1,1000})\x27(?!\w)')
_HYP = re.compile(r'^(?:hypothetically\b|what if\b|suppose\b|imagine (?:that|if|someone|a person)\b|if (?:someone|a person|i) (?:were to|wanted to|said|says|asked)\b)')
_REPORTED = re.compile(r'\b(?:he|she|they|someone|my friend|the character) (?:said|says|wrote|writes|asked)\b')

def review_reason(text):
    t=unicodedata.normalize('NFKC',text).lower().translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"'}))
    t=re.sub(r"\bwon't\b",'will not',t)
    t=re.sub(r"\bcan't\b",'can not',t)
    t=re.sub(r"n't\b",' not',t)
    t=re.sub(r"\bi'm\b",'i am',t)
    t=re.sub(r'[^\S\n]+',' ',t)
    for m in _QUOTE.finditer(t):
        if _TOPIC.search(m.group(1) or m.group(2)):return 'quoted_reference'
    # No match can extend through sentence or contrast-clause boundaries.
    for clause in re.split(r'[.!?;\n]+|\b(?:but|however|although|yet)\b',t):
        clause=clause.strip(' ,:"\x27')
        if _DENIAL.search(clause):return 'explicit_denial'
        if _LIVE.search(clause):return 'desire_to_live'
        if _HYP.search(clause) and _TOPIC.search(clause):return 'hypothetical_reference'
        if _REPORTED.search(clause) and _TOPIC.search(clause):return 'reported_reference'
    return None
