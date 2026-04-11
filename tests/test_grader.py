from models import Category, Priority, EmailAction
from server.grader import grade_easy, grade_medium, grade_hard
from server.email_env_environment import safe_score

EMAIL = {"email_id":"t1","subject":"Payment failed","body":"My payment was declined trying to upgrade my billing plan last night. Please help!","sender":"u@e.com","category":"billing","priority":"high","is_ambiguous":False}

def test_easy_correct():
    a = EmailAction(category=Category.BILLING, priority=Priority.HIGH, is_ambiguous=False)
    assert grade_easy(a, EMAIL) > 0.99

def test_easy_wrong():
    a = EmailAction(category=Category.BUG, priority=Priority.HIGH, is_ambiguous=False)
    assert grade_easy(a, EMAIL) < 0.01

def test_medium_full():
    a = EmailAction(category=Category.BILLING, priority=Priority.HIGH, is_ambiguous=False)
    assert grade_medium(a, EMAIL) > 0.99

def test_medium_partial():
    a = EmailAction(category=Category.BILLING, priority=Priority.MEDIUM, is_ambiguous=False)
    s = grade_medium(a, EMAIL)
    assert 0.5 < s < 0.85

def test_hard_full():
    a = EmailAction(category=Category.BILLING, priority=Priority.HIGH, is_ambiguous=False)
    assert grade_hard(a, EMAIL) > 0.9

def test_safe_score():
    assert safe_score(-1) > 0
    assert safe_score(2) < 1
    assert safe_score(0.5) == 0.5
