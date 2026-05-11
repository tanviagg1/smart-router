"""
RuleBasedClassifier — classifies prompts into task types using keyword
matching, prompt length, and pattern detection.

Task types returned:
  "simple_qa"  — short factual questions
  "code"       — code generation or debugging
  "complex"    — deep reasoning, long explanations, analysis
  "default"    — anything that doesn't match the above

Why rule-based first?
  Fast (no LLM call), deterministic, easy to debug.
  Phase 2 adds an LLM classifier for ambiguous cases.
"""

import re


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

# Substring-safe: these phrases are long enough to not false-match
CODE_PHRASES = {
    "def ", "class ", "import ", "return ", "write a", "implement",
    "algorithm", "function", "data structure", "big o",
    "refactor", "optimize", "debug", "exception", "syntax", "runtime",
    "compile", "regex", "script", "program",
}

# Word-boundary required: short words that would false-match as substrings
# e.g. "api" inside "capital", "graph" inside "graphql"
CODE_WORDS = {
    "code", "api", "sql", "query", "array", "list", "dict", "object",
    "method", "sort", "search", "tree", "graph", "stack", "queue",
    "loop", "error", "binary",
}

def _has_code_signal(text: str) -> list[str]:
    """Check for code keywords using substring for phrases, word-boundary for short words."""
    matched = []
    for kw in CODE_PHRASES:
        if kw in text:
            matched.append(kw)
    for kw in CODE_WORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            matched.append(kw)
    return matched

SIMPLE_QA_KEYWORDS = {
    "what is", "what are", "who is", "who are", "when was", "when did",
    "where is", "where are", "define ", "definition of", "meaning of",
    "how many", "how much", "what does", "which is", "tell me about",
    "what year", "capital of", "how old",
}

COMPLEX_KEYWORDS = {
    "explain", "why does", "how does", "compare", "difference between",
    "pros and cons", "advantages", "disadvantages", "analyze", "analyse",
    "evaluate", "discuss", "elaborate", "in depth", "in detail",
    "step by step", "walk me through", "deep dive", "trade-off",
    "architecture", "design", "philosophy", "implications", "impact",
    "relationship between", "how would you", "what would happen",
}

# Thresholds
SIMPLE_QA_MAX_WORDS = 15
COMPLEX_MIN_WORDS = 40


class RuleBasedClassifier:
    """
    Classifies a prompt into a task type using rules.

    Classification order (first match wins):
    1. Code signals → "code"
    2. Short prompt + simple_qa keywords → "simple_qa"
    3. Long prompt OR complex keywords → "complex"
    4. Short prompt with no matches → "simple_qa" (default for short)
    5. Everything else → "default"

    Usage:
        clf = RuleBasedClassifier()
        result = clf.classify("Write a binary search in Python")
        # → ClassificationResult(task_type="code", confidence=0.9, reason="...")
    """

    def classify(self, prompt: str) -> "ClassificationResult":
        prompt_lower = prompt.lower().strip()
        word_count = len(prompt_lower.split())

        # 1. Code detection — strongest signal (uses word-boundary matching)
        matched_code = _has_code_signal(prompt_lower)
        if matched_code:
            return ClassificationResult(
                task_type="code",
                confidence=0.9,
                reason=f"Code keywords detected: {', '.join(matched_code[:3])}",
                matched_signals=matched_code,
            )

        # 2. Complex — check BEFORE simple_qa so "pros and cons" beats short-prompt rule
        matched_complex = [kw for kw in COMPLEX_KEYWORDS if kw in prompt_lower]
        if matched_complex or word_count >= COMPLEX_MIN_WORDS:
            reason = (
                f"Complexity keywords: {', '.join(matched_complex[:3])}"
                if matched_complex
                else f"Long prompt ({word_count} words suggests detailed request)"
            )
            return ClassificationResult(
                task_type="complex",
                confidence=0.8,
                reason=reason,
                matched_signals=matched_complex,
            )

        # 3. Simple Q&A — short prompt with question pattern
        matched_simple = [kw for kw in SIMPLE_QA_KEYWORDS if prompt_lower.startswith(kw)]
        if matched_simple and word_count <= SIMPLE_QA_MAX_WORDS:
            return ClassificationResult(
                task_type="simple_qa",
                confidence=0.85,
                reason=f"Short factual question detected: '{matched_simple[0]}'",
                matched_signals=matched_simple,
            )

        # 4. Short prompts with no other signals → treat as simple_qa
        if word_count <= SIMPLE_QA_MAX_WORDS:
            return ClassificationResult(
                task_type="simple_qa",
                confidence=0.6,
                reason=f"Short prompt ({word_count} words), no strong signals — assuming simple Q&A",
                matched_signals=[],
            )

        # 5. Default
        return ClassificationResult(
            task_type="default",
            confidence=0.5,
            reason="No strong signals detected — using default model",
            matched_signals=[],
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    """The output of a classifier — task type + confidence + reasoning."""
    task_type: str          # "simple_qa" | "code" | "complex" | "default"
    confidence: float       # 0.0 – 1.0
    reason: str             # human-readable explanation of the decision
    matched_signals: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.task_type.upper()}] confidence={self.confidence:.0%} — {self.reason}"
