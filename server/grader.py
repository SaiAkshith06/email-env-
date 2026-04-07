PRIORITY_ORDER = ["low", "medium", "high", "urgent"]


def grade_easy(action, email):
    if action.category == email["category"]:
        return 1
    return 0


def grade_medium(action, email):
    if action.category == email["category"] and action.priority == email["priority"]:
        return 1
    return 0


def grade_hard(action, email):
    if (action.category == email["category"] and 
        action.priority == email["priority"] and 
        action.is_ambiguous == email["is_ambiguous"]):
        return 1
    return 0