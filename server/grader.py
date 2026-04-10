PRIORITY_ORDER = ["low", "medium", "high", "urgent"]


# ---------------- SAFE SCORE ----------------
def safe_score(score: float) -> float:
    EPS = 1e-6
    try:
        score = float(score)
    except:
        return EPS

    if score <= 0:
        return EPS
    if score >= 1:
        return 1 - EPS

    return score


# ---------------- EASY ----------------
def grade_easy(action, email) -> float:
    score = 1.0 if action.category == email["category"] else 0.0
    return safe_score(score)


# ---------------- MEDIUM ----------------
def grade_medium(action, email) -> float:
    score = 0.0

    # category (0.5)
    if action.category == email["category"]:
        score += 0.5

    # priority (0.5)
    true_p = email["priority"]
    pred_p = action.priority

    if pred_p == true_p:
        score += 0.5
    else:
        try:
            t_idx = PRIORITY_ORDER.index(true_p)
            p_idx = PRIORITY_ORDER.index(pred_p)

            if abs(t_idx - p_idx) == 1:
                score += 0.25
        except:
            pass

    return safe_score(score)


# ---------------- HARD (UPGRADED) ----------------
def grade_hard(action, email) -> float:
    score = 0.0

    # 1. Category (0.35)
    if action.category == email["category"]:
        score += 0.35

    # 2. Priority (0.35)
    true_p = email["priority"]
    pred_p = action.priority

    if pred_p == true_p:
        score += 0.35
    else:
        try:
            t_idx = PRIORITY_ORDER.index(true_p)
            p_idx = PRIORITY_ORDER.index(pred_p)

            if abs(t_idx - p_idx) == 1:
                score += 0.15
        except:
            pass

    # 3. Ambiguity (0.15)
    true_amb = email.get("is_ambiguous", False)
    pred_amb = action.is_ambiguous

    if pred_amb == true_amb:
        score += 0.15
    elif pred_amb and not true_amb:
        score -= 0.1

    # 4. Keyword reasoning bonus (0.15)
    body = email.get("body", "").lower()

    category_keywords = {
        "billing": ["payment", "invoice", "charge", "refund", "billing"],
        "bug": ["crash", "error", "fail", "exception", "broken"],
        "technical": ["login", "account", "access", "setup", "config"],
        "feature": ["feature", "request", "add", "improve", "suggestion"],
        "general": ["question", "help", "info", "inquiry"],
    }

    kws = category_keywords.get(str(action.category), [])

    if any(kw in body for kw in kws):
        score += 0.15

    # prevent negative
    if score < 0:
        score = 0.0

    return safe_score(score)