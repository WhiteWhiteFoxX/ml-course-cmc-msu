def find_modified_max_argmax(L, f):
    F = [f(i) for i in L if type(i) == int]
    if not F: return ()
    m = max(F)
    return m, F.index(m)