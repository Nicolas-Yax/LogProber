def asym_diff(l1,l2):
    """ Returns the difference L1 \ L2 (elements of L1 that aren't in L2).
    No copy of elements L1 is made ! Be careful of pointers ! Plus this function isn't optimised : O(len(l1)*len(l2))"""
    diff = []
    for i in l1:
        not_found = True
        for j in l2:
            if i == j:
                not_found = False
        if not_found:
            diff.append(i)
    return diff