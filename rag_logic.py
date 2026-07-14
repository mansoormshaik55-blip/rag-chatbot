import re

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "who",
    "why",
    "when",
    "where",
    "how",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "and",
    "or",
    "with",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "my",
    "your",
    "our",
    "their",
    "be",
    "being",
    "been",
    "have",
    "has",
    "had",
    "not",
    "if",
    "then",
    "than",
    "about",
    "into",
    "by",
    "as",
}


def _tokenize(text):
    if not text:
        return []

    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()
    if not normalized:
        return []

    return [token for token in normalized.split() if token not in STOPWORDS]


def is_simple_math_question(question):
    if not question:
        return False

    text = question.lower()
    has_digits = bool(re.search(r"\d", text))
    has_math_operator = bool(re.search(r"[+\-*/]", text))

    if has_digits and has_math_operator:
        return True

    math_keywords = [
        "plus",
        "minus",
        "times",
        "multiply",
        "divided",
        "subtract",
        "add",
        "sum",
        "difference",
        "average",
        "percentage",
        "percent",
    ]

    return has_digits and any(keyword in text for keyword in math_keywords)


def should_answer_from_context(question, context):
    if not question or not context:
        return False

    if is_simple_math_question(question):
        return False

    question_tokens = _tokenize(question)
    context_tokens = _tokenize(context)

    if not question_tokens or not context_tokens:
        return False

    overlap = set(question_tokens) & set(context_tokens)
    if not overlap:
        return False

    overlap_ratio = len(overlap) / len(question_tokens)
    return overlap_ratio >= 0.2


def build_grounded_prompt(question, context):
    return f"""
You are a helpful assistant.

Answer ONLY using the information in the provided context.
Do not use outside knowledge.
If the answer is not explicitly supported by the context, reply exactly:
"I couldn't find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}
"""
