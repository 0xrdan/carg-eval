"""Hybrid scoring engine for CARG evaluation.

Scores responses across 5 quality dimensions:
- Accuracy:          LLM-as-judge (falls back to keyword heuristic)
- Relevance:         LLM-as-judge (falls back to keyword heuristic)
- Groundedness:      LLM-as-judge (falls back to attribution heuristic)
- Efficiency:        Always rule-based (word count + latency)
- Format Compliance: Always rule-based (markdown, preamble, length)

When ANTHROPIC_API_KEY is not set, all dimensions use rule-based heuristics.
"""

import os
import re
from dataclasses import dataclass

# Type alias — actual TestCase imported at call site to avoid circular import
# We use duck typing: anything with .route, .expected, .query, .fixture_response

# ---------------------------------------------------------------------------
# Latency targets per route (ms) — from evaluation-methodology.md
# ---------------------------------------------------------------------------

LATENCY_TARGETS = {
    "fast":     {"p50": 2000, "p95": 4000},
    "standard": {"p50": 5000, "p95": 8000},
    "deep":     {"p50": 10000, "p95": 15000},
    "creative": {"p50": 5000, "p95": 8000},
    "research": {"p50": 8000, "p95": 12000},
}

# Word count targets by expected format
WORD_TARGETS = {
    "concise": (10, 200),
    "narrative": (100, 500),
    "structured": (150, 800),
    "creative": (10, 600),
    "academic": (100, 600),
}


# ---------------------------------------------------------------------------
# LLM-as-judge availability
# ---------------------------------------------------------------------------

def has_llm_judge() -> bool:
    """Check if Anthropic API key is available for LLM scoring."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _get_anthropic_client():
    """Lazy-load Anthropic client."""
    import anthropic
    return anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Rule-based scorers (always available)
# ---------------------------------------------------------------------------

def score_efficiency_rule(test_case, response: str, latency_ms: float) -> float:
    """Score efficiency based on word count and latency targets.

    Scoring:
    - 50% weight: word count relative to expected max
    - 50% weight: latency relative to route P50 target
    Returns 0.0-1.0.
    """
    words = len(response.split())
    max_words = test_case.expected.get("max_words", 500)

    # Word count score: 1.0 if under target, decreasing linearly to 0.0 at 2x target
    if words <= max_words:
        word_score = 1.0
    elif words <= max_words * 2:
        word_score = 1.0 - (words - max_words) / max_words
    else:
        word_score = 0.0

    # Latency score: 1.0 if under P50, scaling down to 0.0 at 2x P95
    targets = LATENCY_TARGETS.get(test_case.route, {"p50": 5000, "p95": 8000})
    p50 = targets["p50"]
    p95 = targets["p95"]

    if latency_ms <= p50:
        latency_score = 1.0
    elif latency_ms <= p95:
        latency_score = 1.0 - 0.5 * (latency_ms - p50) / (p95 - p50)
    elif latency_ms <= p95 * 2:
        latency_score = 0.5 - 0.5 * (latency_ms - p95) / p95
    else:
        latency_score = 0.0

    return round(0.5 * word_score + 0.5 * latency_score, 3)


def score_format_compliance_rule(test_case, response: str) -> float:
    """Score format compliance based on structural checks.

    Checks:
    - Word count within expected range
    - Markdown presence (for structured/academic formats)
    - No generic preamble ("As an AI...", "I'd be happy to...")
    - Length within bounds
    Returns 0.0-1.0.
    """
    expected_format = test_case.expected.get("expected_format", "narrative")
    word_range = WORD_TARGETS.get(expected_format, (10, 500))
    words = len(response.split())

    checks_passed = 0
    total_checks = 4

    # 1. Word count within expected range
    min_words, max_words_fmt = word_range
    if min_words <= words <= max_words_fmt * 1.2:  # 20% tolerance on upper bound
        checks_passed += 1

    # 2. Markdown presence check (for structured/academic)
    has_markdown = bool(re.search(r'(\*\*|##|```|\|.*\|)', response))
    if expected_format in ("structured", "academic"):
        if has_markdown:
            checks_passed += 1
    else:
        # For other formats, markdown is fine but not required
        checks_passed += 1

    # 3. No generic AI preamble
    preamble_patterns = [
        r'^(As an AI|I\'d be happy to|Sure!|Of course!|Certainly!)',
        r'^(Great question|That\'s a great question)',
    ]
    has_preamble = any(
        re.search(p, response.strip(), re.IGNORECASE)
        for p in preamble_patterns
    )
    if not has_preamble:
        checks_passed += 1

    # 4. Not empty and not absurdly long
    if 1 <= words <= 2000:
        checks_passed += 1

    return round(checks_passed / total_checks, 3)


def score_accuracy_rule(test_case, response: str) -> float:
    """Heuristic accuracy score based on expected keyword presence.

    Checks should_mention and should_not_mention lists.
    Returns 0.0-1.0.
    """
    should_mention = test_case.expected.get("should_mention", [])
    should_not_mention = test_case.expected.get("should_not_mention", [])

    if not should_mention and not should_not_mention:
        return 0.7  # Neutral score when no keywords defined

    response_lower = response.lower()
    total = len(should_mention) + len(should_not_mention)
    hits = 0

    for keyword in should_mention:
        if keyword.lower() in response_lower:
            hits += 1

    for keyword in should_not_mention:
        if keyword.lower() not in response_lower:
            hits += 1

    return round(hits / total, 3) if total > 0 else 0.7


def score_relevance_rule(test_case, response: str) -> float:
    """Heuristic relevance score based on query-response term overlap.

    Uses prefix matching (poor-man's stemming) to handle morphological
    variants (e.g. "technologies" matches "technology"). Gives a higher
    floor for very short queries where exact overlap is unreliable.
    Returns 0.0-1.0.
    """
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "so", "if", "than", "too", "very", "just", "about", "what",
        "how", "when", "where", "who", "which", "this", "that", "these",
        "those", "it", "its", "he", "she", "they", "them", "his", "her",
        "their", "my", "your", "our", "dan", "dan's",
    }

    query_words = set(re.findall(r'\b\w+\b', test_case.query.lower())) - stop_words
    response_words = set(re.findall(r'\b\w+\b', response.lower())) - stop_words

    if not query_words:
        return 0.7

    # Prefix matching: "technologies" matches "technology", "tech", etc.
    def _prefix_match(query_word: str, response_set: set) -> bool:
        stem = query_word[:4] if len(query_word) > 4 else query_word[:3]
        return any(rw.startswith(stem) or query_word.startswith(rw[:4] if len(rw) > 4 else rw[:3])
                   for rw in response_set)

    # Count matches using both exact and prefix matching
    matches = 0
    for qw in query_words:
        if qw in response_words or _prefix_match(qw, response_words):
            matches += 1

    coverage = matches / len(query_words)

    # Bonus for substantial response (not just echoing keywords)
    depth_bonus = min(0.3, len(response_words) / 300 * 0.3) if coverage > 0.3 else 0.0

    # Floor for very short queries (<=2 content words) where overlap is unreliable
    if len(query_words) <= 2 and len(response_words) > 20:
        coverage = max(coverage, 0.5)

    return round(min(1.0, coverage + depth_bonus), 3)


def score_groundedness_rule(test_case, response: str) -> float:
    """Heuristic groundedness score based on attribution signals.

    Checks for:
    - Attribution phrases ("according to", "based on", etc.)
    - Hedging language ("the article mentions", etc.)
    - Absence of ungrounded claims (superlatives, certainty without evidence)
    Returns 0.0-1.0.
    """
    if not test_case.expected.get("grounding_required", True):
        return 0.8  # Lenient for creative responses

    response_lower = response.lower()

    # Attribution signals
    attribution_patterns = [
        r'based on', r'according to', r'the article', r'dan[\'\u2019]?s?\s+(?:portfolio|work|experience|system)',
        r'the (?:carg|chatbot|system|pipeline)', r'his (?:work|experience|portfolio)',
        r'documented', r'described', r'mentions',
    ]
    attribution_count = sum(
        1 for p in attribution_patterns
        if re.search(p, response_lower)
    )

    # Ungrounded signals (penalize)
    ungrounded_patterns = [
        r'(?:definitely|certainly|obviously|clearly) the best',
        r'no doubt', r'everyone knows', r'it\'s clear that',
    ]
    ungrounded_count = sum(
        1 for p in ungrounded_patterns
        if re.search(p, response_lower)
    )

    # Score: base 0.5 + attribution bonus - ungrounded penalty
    attribution_score = min(0.5, attribution_count * 0.1)
    ungrounded_penalty = min(0.3, ungrounded_count * 0.15)

    return round(min(1.0, max(0.0, 0.5 + attribution_score - ungrounded_penalty)), 3)


# ---------------------------------------------------------------------------
# LLM-as-judge scorers
# ---------------------------------------------------------------------------

def _llm_judge_score(system_prompt: str, user_prompt: str) -> float:
    """Call Claude Haiku as a judge. Returns a 0.0-1.0 score.

    The prompt must instruct the model to return ONLY a JSON object
    with a "score" field (0-10) and a "reasoning" field.
    """
    client = _get_anthropic_client()

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()

        # Extract score from JSON response
        import json
        # Handle possible markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        data = json.loads(text)
        raw_score = float(data["score"])
        return round(max(0.0, min(1.0, raw_score / 10.0)), 3)
    except Exception:
        # Fallback to rule-based on any LLM failure
        return -1.0  # Sentinel: caller should use rule-based fallback


def score_accuracy_llm(test_case, response: str) -> float:
    """LLM-as-judge for factual accuracy."""
    system = (
        "You are an evaluation judge. Score the factual accuracy of an AI chatbot's "
        "response about a person named Dan and his portfolio/work. "
        "Return ONLY a JSON object: {\"score\": <0-10>, \"reasoning\": \"<brief>\"}\n"
        "Scoring guide:\n"
        "10: All facts correct and well-supported\n"
        "7-9: Mostly correct, minor omissions\n"
        "4-6: Some correct info, some inaccuracies or gaps\n"
        "1-3: Mostly inaccurate or very incomplete\n"
        "0: Completely wrong or irrelevant"
    )
    keywords = test_case.expected.get("should_mention", [])
    keyword_hint = f"\nExpected to mention: {', '.join(keywords)}" if keywords else ""

    user = (
        f"Query: {test_case.query}\n\n"
        f"Response:\n{response}\n\n"
        f"Route: {test_case.route}{keyword_hint}\n\n"
        "Score the factual accuracy (0-10)."
    )
    return _llm_judge_score(system, user)


def score_relevance_llm(test_case, response: str) -> float:
    """LLM-as-judge for query-response relevance."""
    system = (
        "You are an evaluation judge. Score how well the response addresses the "
        "user's actual query intent. "
        "Return ONLY a JSON object: {\"score\": <0-10>, \"reasoning\": \"<brief>\"}\n"
        "Scoring guide:\n"
        "10: Directly and completely addresses the query\n"
        "7-9: Addresses the query well with minor tangents\n"
        "4-6: Partially addresses the query\n"
        "1-3: Mostly off-topic\n"
        "0: Completely irrelevant"
    )
    user = (
        f"Query: {test_case.query}\n\n"
        f"Response:\n{response}\n\n"
        "Score the relevance (0-10)."
    )
    return _llm_judge_score(system, user)


def score_groundedness_llm(test_case, response: str) -> float:
    """LLM-as-judge for groundedness (claims traceable to context)."""
    system = (
        "You are an evaluation judge. Score how well the response's claims are "
        "grounded — traceable to the context about Dan's work and portfolio. "
        "Penalize hallucinated claims or unsupported statements. "
        "Return ONLY a JSON object: {\"score\": <0-10>, \"reasoning\": \"<brief>\"}\n"
        "Scoring guide:\n"
        "10: All claims clearly grounded, good attribution\n"
        "7-9: Most claims grounded, minor unattributed statements\n"
        "4-6: Mix of grounded and ungrounded claims\n"
        "1-3: Mostly ungrounded or hallucinated\n"
        "0: Pure hallucination"
    )
    grounding_note = ""
    if not test_case.expected.get("grounding_required", True):
        grounding_note = "\nNote: This is a creative/open-ended query — lighter grounding expectations."

    user = (
        f"Query: {test_case.query}\n\n"
        f"Response:\n{response}\n\n"
        f"Route: {test_case.route}, Category: {test_case.category}{grounding_note}\n\n"
        "Score the groundedness (0-10)."
    )
    return _llm_judge_score(system, user)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def score_result(test_case, response: str, latency_ms: float):
    """Score a response across all 5 dimensions.

    Uses LLM-as-judge for accuracy/relevance/groundedness when available,
    falling back to rule-based heuristics.
    """
    # Import here to avoid circular dependency
    from eval import Scores

    # Rule-based dimensions (always used)
    efficiency = score_efficiency_rule(test_case, response, latency_ms)
    format_compliance = score_format_compliance_rule(test_case, response)

    # Accuracy: LLM or rule-based
    if has_llm_judge():
        accuracy = score_accuracy_llm(test_case, response)
        if accuracy < 0:  # LLM failed, fallback
            accuracy = score_accuracy_rule(test_case, response)
    else:
        accuracy = score_accuracy_rule(test_case, response)

    # Relevance: LLM or rule-based
    if has_llm_judge():
        relevance = score_relevance_llm(test_case, response)
        if relevance < 0:
            relevance = score_relevance_rule(test_case, response)
    else:
        relevance = score_relevance_rule(test_case, response)

    # Groundedness: LLM or rule-based
    if has_llm_judge():
        groundedness = score_groundedness_llm(test_case, response)
        if groundedness < 0:
            groundedness = score_groundedness_rule(test_case, response)
    else:
        groundedness = score_groundedness_rule(test_case, response)

    return Scores(
        accuracy=accuracy,
        relevance=relevance,
        groundedness=groundedness,
        efficiency=efficiency,
        format_compliance=format_compliance,
    )
