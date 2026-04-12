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

    kws = category_keywords.get(email["category"], [])

    if any(kw in body for kw in kws):
        score += 0.15

    # Check priority-signal consistency
    urgent_signals = ["production", "down", "data loss", "breach", "critical", "all users"]
    high_signals = ["broken", "failed", "cannot", "blocked", "payment"]
    body_lower = body

    has_urgent_signal = any(s in body_lower for s in urgent_signals)
    has_high_signal = any(s in body_lower for s in high_signals)

    if action.priority == "urgent" and has_urgent_signal and email["priority"] == "urgent":
        score += 0.05    # correctly identified urgency signal
    elif action.priority == "urgent" and not has_urgent_signal:
        score -= 0.05    # hallucinated urgency

    # prevent negative
    if score < 0:
        score = 0.0

    return safe_score(score)


# ---------------- ADAPTIVE ----------------
def grade_adaptive(action, email, investigate_count=0) -> float:
    # Start with hard score
    score = float(grade_hard(action, email))
    
    if investigate_count == 0 and score > 0.85:
        score += 0.05    # got it right without needing a hint
    elif investigate_count == 1 and score > 0.85:
        pass            # normal - used investigate once, got it right
    elif investigate_count > 1:
        score -= 0.1 * (investigate_count - 1)    # penalise over-investigating

    # SLA bonus
    hours = email.get("hours_since_received", 0)
    if action.priority == "urgent" and hours > 48 and email["priority"] == "urgent":
        score += 0.05    # correctly escalated an old urgent email
        
    return safe_score(max(0.0, score))