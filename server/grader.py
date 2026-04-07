PRIORITY_ORDER = ["low", "medium", "high", "urgent"]


def grade_easy(action, email):
    score = 1.0 if action.category == email["category"] else 0.0
    return max(0.01, min(0.99, float(score)))


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
        except ValueError:
            pass
            
    return max(0.01, min(0.99, float(score)))


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
        
    return max(0.01, min(0.99, float(score)))