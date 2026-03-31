dic = dict()
S = set()
for line in open("info.csv", "r").readlines():
    line = line.strip("\n")
    acid, labels = line.split(":")
    labels = labels.split(",")
    dic[acid] = labels
    for x in labels:
        S.add(x)

l_to_id = {x:i for i, x in enumerate(S)}
with open("tag.csv", "w") as f:
    for k, v in l_to_id.items():
        print(f"{k},{v}", file=f)

with open("output.csv", "w") as f:
    for acid, labels in dic.items():
        ids = [str(l_to_id[l]) for l in labels]
        moji = acid + ":" + ",".join(ids)
        print(moji, file=f)

