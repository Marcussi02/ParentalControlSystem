import re
import numpy as np

# ---------------------------------------------------------------------------
# Keyword lists per category.
# Single words use word-boundary matching; phrases use substring matching.
# ---------------------------------------------------------------------------

_CATEGORIES = {
    "SEXUAL": {
        "score": 50,
        "label": "Sexual/adult content",
        "words": [
            # Explicit
            "sex", "sexy", "sexual", "nude", "naked", "nudity", "nsfw", "porn",
            "pornography", "erotic", "xxx", "explicit", "adult content", "18+",
            "onlyfans", "only fans", "hookup", "hook up", "one night stand",
            "boobs", "cleavage", "genitals",
            "strip tease", "striptease", "lingerie", "bikini try on",
            "body reveal", "thirst trap", "naughty",
            "rated r", "mature content", "mature audiences",
            # Suggestive / borderline
            "making out", "getting laid", "body count", "sleep with",
            "sleeping with", "had sex", "sexual tension", "seductive",
            "provocative", "sensual", "bedroom challenge",
        ],
    },
    "VIOLENCE": {
        "score": 35,
        "label": "Violent or graphic content",
        "words": [
            # Graphic — require fairly explicit language
            "murder", "gore", "decapitation", "torture", "execution",
            "graphic violence", "brutal killing", "massacre", "beheading",
            "real death", "dead body", "corpse", "stabbing", "stabbed",
            "shot dead", "bloody", "brutal fight", "beaten to death",
            "graphic injury", "real accident", "death footage",
            # Broader but still intentional violence signals
            "street fight", "school fight", "brawl", "assault",
            "knocked out", "ko'd", "jumped", "mugging",
            "shooting", "gunshot", "gun fight", "war footage", "combat footage",
            "gets killed", "how to hurt", "how to fight", "how to attack",
        ],
    },
    "WEAPONS": {
        "score": 25,
        "label": "Weapons or dangerous items",
        "words": [
            "pistol", "rifle", "shotgun", "revolver", "firearm",
            "ak47", "ak-47", "ar-15", "machine gun",
            "switchblade", "brass knuckles",
            "explosive", "bomb", "grenade", "molotov",
            "buying a gun", "gun review", "gun unboxing",
            "how to make a weapon", "homemade weapon", "illegal weapon",
        ],
    },
    "DRUGS": {
        "score": 30,
        "label": "Drug or alcohol content",
        "words": [
            # Hard drugs
            "cocaine", "heroin", "methamphetamine", "meth", "mdma",
            "lsd", "acid trip", "lsd trip", "ecstasy", "molly",
            "ketamine", "fentanyl", "crack", "snorting", "shooting up",
            "overdose", "drug trip", "rolling on",
            # Cannabis
            "marijuana", "weed", "cannabis", "weed haul", "drug haul",
            "getting high", "smoke weed", "stoned", "blazed",
            # Alcohol (requires more specific phrasing — "drunk" alone is too common)
            "getting drunk", "drunk challenge", "drinking challenge",
            "alcohol challenge", "shots challenge", "blackout drunk", "wasted",
            # Vaping / smoking
            "vaping", "vape tricks", "smoking challenge",
            "nicotine challenge", "juul",
        ],
    },
    "PROFANITY": {
        "score": 15,
        "label": "Strong profanity",
        "words": [
            # Only strong profanity — mild terms removed (too many false positives)
            "fuck", "fucking", "fucked", "fucker", "f-word",
            "shit", "bullshit", "piece of shit",
            "asshole", "bitch", "bastard",
            "cunt", "motherfucker", "son of a bitch",
        ],
    },
    "HORROR": {
        "score": 20,
        "label": "Disturbing or horror content",
        "words": [
            "demonic possession", "exorcism", "satanic ritual", "666",
            "cursed video", "creepy pasta", "creepypasta",
            "this video will", "watch at your own risk",
            "not for the faint", "disturbing footage", "disturbing video",
            "graphic content warning", "viewer discretion",
            "scariest video", "most disturbing", "nightmare fuel",
            "ghost caught on camera", "demon caught on camera",
            "ouija board", "dark web video", "deep web", "nsfl",
        ],
    },
    "DANGEROUS_CHALLENGE": {
        "score": 30,
        "label": "Dangerous viral challenge",
        "words": [
            "blackout challenge", "choking challenge", "fire challenge",
            "skull breaker", "skull-breaker", "tide pod", "tide pods",
            "salt and ice challenge", "benadryl challenge", "nyquil chicken",
            "devious lick", "outlet challenge",
            "car surfing", "train surfing",
            "dangerous stunt", "life-threatening challenge",
        ],
    },
    "GAMBLING": {
        "score": 25,
        "label": "Gambling content",
        "words": [
            "casino", "slot machine", "gambling", "poker game",
            "blackjack", "roulette", "sports betting", "placing bets",
            "won at casino", "jackpot win", "gambling addiction",
            "online casino", "loot boxes", "csgo gambling", "skin gambling",
        ],
    },
    "SELF_HARM": {
        "score": 60,
        "label": "Self-harm or dangerous mental health content",
        "words": [
            "suicide", "suicidal", "self harm", "self-harm", "cutting myself",
            "anorexia", "bulimia", "pro ana", "starving myself",
            "how to end it", "want to die", "kill myself",
            "ending my life", "not worth living", "no reason to live",
            "thinspo", "thinspiration", "pro mia",
        ],
    },
    "HATE": {
        "score": 45,
        "label": "Hate speech or discrimination",
        "words": [
            "white supremacy", "white power", "ethnic cleansing",
            "nazi propaganda", "racial slur", "homophobic", "transphobic",
            "islamophobia", "antisemitic", "hate group",
            "white nationalist", "great replacement", "race war",
        ],
    },
    "PREDATORY": {
        "score": 55,
        "label": "Predatory or grooming-related content",
        "words": [
            "grooming", "predator", "meet in person", "meet irl",
            "secret meeting", "don't tell your parents",
            "online predator", "catfish", "catfishing",
            "child predator", "inappropriate contact",
        ],
    },
}

# Skin detection thresholds in YCrCb color space
_SKIN_CR_MIN, _SKIN_CR_MAX = 133, 173
_SKIN_CB_MIN, _SKIN_CB_MAX = 77, 127
_SKIN_Y_MIN = 80
_SKIN_RATIO_ALERT = 0.32   # >32% skin-colored pixels may indicate nudity

# Blood/gore: very red pixels relative to green and blue
_BLOOD_R_MIN = 150
_BLOOD_R_MINUS_G = 62
_BLOOD_R_MINUS_B = 55
_BLOOD_RATIO_ALERT = 0.06  # >6% blood-red pixels in a thumbnail


def _compile_patterns(words: list) -> list:
    patterns = []
    for w in words:
        if " " in w:
            patterns.append(re.compile(re.escape(w), re.IGNORECASE))
        else:
            patterns.append(re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE))
    return patterns


_COMPILED = {cat: _compile_patterns(data["words"]) for cat, data in _CATEGORIES.items()}


class ContentFilter:
    """
    Checks a YouTube video (title + description + thumbnail + age_limit)
    for content inappropriate for children.

    Title matches are weighted with a bonus since creators deliberately choose
    titles. Description-only hits score at face value.

    Returns safety_score (0 = safe, 100 = very inappropriate).
    """

    # Extra points when the keyword is found in the title specifically
    _TITLE_BONUS = 10

    def analyze(self, title: str, description: str = "",
                thumbnail_img_bgr=None, age_limit: int = 0) -> dict:
        title_lower = title.lower()
        desc_lower = description.lower()

        title_hits: dict[str, list] = {}
        categories_flagged: list[str] = []
        flags: list[str] = []
        raw_score = 0

        # --- Age restriction (strongest signal) ---
        if age_limit and age_limit > 0:
            categories_flagged.append("AGE_RESTRICTED")
            flags.append(f"YouTube age-restricted (limit: {age_limit}+)")
            raw_score += 65

        # --- Text analysis ---
        for cat, data in _CATEGORIES.items():
            title_matches = []
            desc_matches = []
            for pattern in _COMPILED[cat]:
                if pattern.search(title_lower):
                    title_matches.extend(pattern.findall(title_lower))
                elif pattern.search(desc_lower):
                    desc_matches.extend(pattern.findall(desc_lower))

            all_matches = title_matches + desc_matches
            if all_matches:
                unique = list(dict.fromkeys(m.lower() for m in all_matches))
                title_hits[cat] = unique
                categories_flagged.append(cat)
                raw_score += data["score"]
                if title_matches:
                    raw_score += self._TITLE_BONUS
                flags.append(f"{data['label']}: {', '.join(unique[:4])}")

        # --- Thumbnail visual analysis ---
        thumbnail_flags: list[str] = []
        if thumbnail_img_bgr is not None:
            thumb_flags, visual_score = self._analyze_thumbnail(thumbnail_img_bgr)
            thumbnail_flags = thumb_flags
            for tf in thumb_flags:
                flags.append(tf)
            raw_score += visual_score

        safety_score = min(100, raw_score)
        is_inappropriate = safety_score >= 25

        return {
            "safety_score": safety_score,
            "is_inappropriate": is_inappropriate,
            "categories_flagged": categories_flagged,
            "flags": flags,
            "title_hits": title_hits,
            "thumbnail_flags": thumbnail_flags,
        }

    def _analyze_thumbnail(self, img_bgr: np.ndarray) -> tuple[list, int]:
        """Returns (flag_strings, total_score_addition)."""
        flags = []
        score = 0
        try:
            import cv2

            # Skin exposure
            ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin_mask = (
                (y > _SKIN_Y_MIN) &
                (cr >= _SKIN_CR_MIN) & (cr <= _SKIN_CR_MAX) &
                (cb >= _SKIN_CB_MIN) & (cb <= _SKIN_CB_MAX)
            )
            skin_ratio = float(np.sum(skin_mask)) / skin_mask.size
            if skin_ratio > _SKIN_RATIO_ALERT:
                flags.append(
                    f"High skin exposure in thumbnail ({skin_ratio:.0%}) — possible nudity"
                )
                score += min(40, int(20 + (skin_ratio - _SKIN_RATIO_ALERT) * 150))

            # Blood / gore
            b, g, r = (img_bgr[:, :, 0].astype(int),
                       img_bgr[:, :, 1].astype(int),
                       img_bgr[:, :, 2].astype(int))
            blood_mask = (
                (r > _BLOOD_R_MIN) &
                ((r - g) > _BLOOD_R_MINUS_G) &
                ((r - b) > _BLOOD_R_MINUS_B)
            )
            blood_ratio = float(np.sum(blood_mask)) / blood_mask.size
            if blood_ratio > _BLOOD_RATIO_ALERT:
                flags.append(
                    f"High red-dominant pixels in thumbnail ({blood_ratio:.0%})"
                    " — possible violence/gore"
                )
                score += min(30, int(15 + (blood_ratio - _BLOOD_RATIO_ALERT) * 150))

        except Exception:
            pass
        return flags, score
