"""
Tests for RuleBasedClassifier.

Run: pytest tests/test_classifier.py -v
"""

import pytest
from router.classifier import RuleBasedClassifier


@pytest.fixture
def clf():
    return RuleBasedClassifier()


class TestCodeClassification:

    def test_write_function_is_code(self, clf):
        result = clf.classify("Write a binary search algorithm in Python")
        assert result.task_type == "code"

    def test_implement_keyword_is_code(self, clf):
        result = clf.classify("Implement a linked list in JavaScript")
        assert result.task_type == "code"

    def test_debug_is_code(self, clf):
        result = clf.classify("Debug this function: def foo(x): return x/0")
        assert result.task_type == "code"

    def test_def_keyword_is_code(self, clf):
        result = clf.classify("def calculate_sum(numbers): return sum(numbers)")
        assert result.task_type == "code"

    def test_sql_query_is_code(self, clf):
        result = clf.classify("Write a SQL query to find duplicate rows")
        assert result.task_type == "code"

    def test_code_confidence_is_high(self, clf):
        result = clf.classify("Write a sort algorithm")
        assert result.confidence >= 0.85


class TestSimpleQAClassification:

    def test_what_is_is_simple(self, clf):
        result = clf.classify("What is recursion?")
        assert result.task_type == "simple_qa"

    def test_who_is_is_simple(self, clf):
        result = clf.classify("Who invented Python?")
        assert result.task_type == "simple_qa"

    def test_define_is_simple(self, clf):
        result = clf.classify("Define polymorphism")
        assert result.task_type == "simple_qa"

    def test_short_factual_is_simple(self, clf):
        result = clf.classify("What is the capital of France?")
        assert result.task_type == "simple_qa"

    def test_how_many_is_simple(self, clf):
        result = clf.classify("How many bits in a byte?")
        assert result.task_type == "simple_qa"


class TestComplexClassification:

    def test_explain_in_depth_is_complex(self, clf):
        result = clf.classify("Explain the CAP theorem in depth")
        assert result.task_type == "complex"

    def test_compare_is_complex(self, clf):
        result = clf.classify("Compare microservices vs monolithic architecture")
        assert result.task_type == "complex"

    def test_long_prompt_is_complex(self, clf):
        prompt = " ".join(["word"] * 50)  # 50 words — over COMPLEX_MIN_WORDS threshold
        result = clf.classify(prompt)
        assert result.task_type == "complex"

    def test_pros_and_cons_is_complex(self, clf):
        result = clf.classify("What are the pros and cons of using microservices?")
        assert result.task_type == "complex"

    def test_trade_off_is_complex(self, clf):
        result = clf.classify("What are the trade-offs between REST and GraphQL?")
        assert result.task_type == "complex"


class TestClassificationResult:

    def test_result_has_reason(self, clf):
        result = clf.classify("What is Python?")
        assert result.reason != ""

    def test_result_str_contains_task_type(self, clf):
        result = clf.classify("Write a function")
        assert result.task_type.upper() in str(result)

    def test_confidence_between_0_and_1(self, clf):
        for prompt in ["What is X?", "Write code", "Explain deeply why this works"]:
            result = clf.classify(prompt)
            assert 0.0 <= result.confidence <= 1.0

    def test_code_beats_simple_qa_for_code_prompt(self, clf):
        # Even if prompt starts with "what", code keyword wins
        result = clf.classify("What is the code for a binary search algorithm?")
        assert result.task_type == "code"
