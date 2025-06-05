def check(s, filename):
    f = open(filename, 'w')
    s = s.lower().split()
    set_s = set(s)
    dct = dict()
    for word in set_s:
        dct[word] = s.count(word)
    sorted_dct = dict(sorted(dct.items()))
    for key in sorted_dct:
        f.write(key + ' ' + str(sorted_dct[key]) + '\n')
    f.close()