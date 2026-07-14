from rag_logic import should_answer_from_context

assert should_answer_from_context("Who won the 2024 World Cup?", "The report describes the company's revenue growth in 2023.") is False
assert should_answer_from_context("What was the company's revenue growth in 2023?", "The company reported revenue growth of 18% in 2023.") is True
assert should_answer_from_context("What is 2 + 2?", "The document explains the company's quarterly revenue trends.") is False
print("Grounding checks passed")
