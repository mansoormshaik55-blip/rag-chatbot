from rag_logic import should_answer_from_context


def test_returns_false_when_question_has_no_overlap_with_context():
    question = "Who won the 2024 World Cup?"
    context = "The report describes the company's revenue growth in 2023."

    assert should_answer_from_context(question, context) is False


def test_returns_true_when_question_has_overlap_with_context():
    question = "What was the company's revenue growth in 2023?"
    context = "The company reported revenue growth of 18% in 2023."

    assert should_answer_from_context(question, context) is True


def test_returns_false_for_simple_math_question():
    question = "What is 2 + 2?"
    context = "The document explains the company's quarterly revenue trends."

    assert should_answer_from_context(question, context) is False
