
def select_max(iterable, key=lambda x: x):
    candidate = None
    candidate_val = None
    for x in iterable:
        val = key(x)
        if candidate_val is None or val > candidate_val:
            candidate_val = val
            candidate = x
    return candidate

def select_min(iterable, key=lambda x: x):
    candidate = None
    candidate_val = None
    for x in iterable:
        val = key(x)
        if candidate_val is None or val < candidate_val:
            candidate_val = val
            candidate = x
    return candidate
