PRIORITY_ORDER = ["low", "medium", "high", "urgent"]


def grade_easy(action, email):
    # Only category matters
    return 1.0 if action.category == email["category"] else 0.0


def grade_medium(action, email):
    score = 0.0

    # Category (important)
    if action.category == email["category"]:
        score += 0.5

    # Priority with tolerance
    ai = PRIORITY_ORDER.index(action.priority)
    ei = PRIORITY_ORDER.index(email["priority"])

    if ai == ei:
        score += 0.5
    elif abs(ai - ei) == 1:
        score += 0.25  # close guess

    return score


def grade_hard(action, email):
    score = 0.0

    # Category
    if action.category == email["category"]:
        score += 0.4

    # Priority
    if action.priority == email["priority"]:
        score += 0.4

    # Ambiguity (IMPORTANT FIX)
    if action.is_ambiguous == email["is_ambiguous"]:
        score += 0.2
    elif action.is_ambiguous and not email["is_ambiguous"]:
        score -= 0.1  # penalty for always guessing True

    return max(0.0, min(1.0, score))