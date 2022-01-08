def calcAccuracy(target, predict):
    dict = {i : [] for i in set(target)}

    for t, p in zip(target, predict):
        dict[t].append(1 if t == p else 0)

    for i in dict.keys():
        dict[i] = sum(dict[i]) / len(dict[i])

    return dict
