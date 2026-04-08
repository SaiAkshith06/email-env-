PRIORITY_ORDER = ["low", "medium", "high", "urgent"]

# ---------------- SAFE SCORE ----------------
def safe_score(score):
    EPS = 1e-6
    try:
        score = float(score)
    except:
        return EPS

    # HARD clamp BEFORE anything else
    if score <= 0:
        return EPS
    if score >= 1:
        return 1 - EPS

    # extra protection (prevents rounding issues)
    if score < EPS:
        return EPS
    if score > 1 - EPS:
        return 1 - EPS

    return score


# ---------------- EASY ----------------
def grade_easy(action, email):
    score = 1.0 if action.category == email["category"] else 0.0
    return safe_score(score)


# ---------------- MEDIUM ----------------
def grade_medium(action, email):
    score = 0.0

    if action.category == email["category"]:
        score += 0.5

    true_priority = email["priority"]
    pred_priority = action.priority

    if pred_priority == true_priority:
        score += 0.5
    else:
        try:
            true_idx = PRIORITY_ORDER.index(true_priority)
            pred_idx = PRIORITY_ORDER.index(pred_priority)

            if abs(true_idx - pred_idx) == 1:
                score += 0.25
        except Exception:
            pass

    return safe_score(score)


# ---------------- HARD ----------------
def grade_hard(action, email):
    score = 0.0

    if action.category == email["category"]:
        score += 0.4

    if action.priority == email["priority"]:
        score += 0.4

    true_amb = email.get("is_ambiguous", False)
    pred_amb = action.is_ambiguous

    if pred_amb == true_amb:
        score += 0.2
    elif pred_amb and not true_amb:
        score -= 0.1

    # IMPORTANT: prevent negative BEFORE safe_score
    if score < 0:
        score = 0.0

    return safe_score(score)