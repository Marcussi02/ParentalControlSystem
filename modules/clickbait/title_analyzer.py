import re

# Kids-targeted clickbait keywords (matched case-insensitively)
KIDS_CLICKBAIT_KEYWORDS = [
    # Challenge / stunt bait
    "challenge", "gone wrong", "gone right", "gone crazy", "went wrong",
    # Superlative / shock bait
    "amazing", "shocking", "unbelievable", "impossible", "incredible",
    "insane", "mind blowing", "mind-blowing", "unreal", "epic fail",
    # Secret / mystery bait
    "secret", "hidden", "revealed", "exposing", "exposed", "the truth",
    "you won't believe", "you wont believe", "they don't want you",
    # Reaction bait
    "omg", "oh my god", "oh my gosh", "wait for it", "watch till the end",
    "watch to the end", "wait until the end", "stay till the end",
    # Fear / danger bait
    "dangerous", "scary", "haunted", "worst ever", "near death",
    # Forbidden / exclusive bait
    "banned", "illegal", "not allowed", "no one knows",
    # Reward bait
    "free", "win", "giveaway", "prize", "give away",
    # Superlative size / rank bait
    "biggest", "largest", "smallest", "most expensive", "cheapest",
    "world record", "world's", "number one", "#1",
    # Time-pressure bait
    "24 hours", "overnight", "for a week", "for 30 days",
    "eating only", "living on", "surviving on",
    # Cliffhanger / sequel bait
    "part 2", "part 3", "the end", "finally", "last ever",
    # Trust-buster bait (ironic signal)
    "no clickbait", "not clickbait", "100% real", "not fake",
]

EXCESSIVE_EMOJI_THRESHOLD = 3
CAPS_RATIO_THRESHOLD = 0.5

# Unicode ranges that cover common emoji blocks
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class TitleAnalyzer:
    """Analyzes a YouTube title for kids-targeted clickbait signals."""

    def analyze(self, title: str, kids_mode_multiplier: float = 1.0) -> dict:
        """
        Returns sub_score (0–40), matched signals, and detail metrics.
        kids_mode_multiplier: apply 1.2 when kids mode is on.
        """
        title_lower = title.lower()
        matched_keywords = self._find_keywords(title_lower)
        caps_ratio = self._caps_ratio(title)
        exclamation_count = title.count("!")
        question_count = title.count("?")
        emoji_count = self._count_emoji(title)

        points = 0
        flags = []

        # Keyword scoring (capped at 20 pts)
        keyword_pts = min(len(matched_keywords) * 5, 20)
        points += keyword_pts
        if matched_keywords:
            flags.append(f"Keywords: {', '.join(matched_keywords[:5])}")

        # ALL CAPS
        if caps_ratio > CAPS_RATIO_THRESHOLD:
            points += 8
            flags.append(f"Excessive caps ({caps_ratio:.0%})")

        # Punctuation
        if exclamation_count > 2:
            points += 4
            flags.append(f"{exclamation_count} exclamation marks")

        if question_count > 1:
            points += 3
            flags.append(f"{question_count} question marks")

        # Emoji
        if emoji_count > EXCESSIVE_EMOJI_THRESHOLD:
            points += 5
            flags.append(f"{emoji_count} emoji")

        raw_score = min(40, points)
        sub_score = min(40, int(raw_score * kids_mode_multiplier))

        return {
            "sub_score": sub_score,
            "caps_ratio": caps_ratio,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "emoji_count": emoji_count,
            "matched_keywords": matched_keywords,
            "flags": flags,
        }

    def _count_emoji(self, text: str) -> int:
        return sum(len(m.group()) for m in _EMOJI_RE.finditer(text))

    def _caps_ratio(self, text: str) -> float:
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return 0.0
        upper = sum(1 for c in alpha if c.isupper())
        return upper / len(alpha)

    def _find_keywords(self, title_lower: str) -> list:
        found = []
        for kw in KIDS_CLICKBAIT_KEYWORDS:
            if kw in title_lower and kw not in found:
                found.append(kw)
        return found
