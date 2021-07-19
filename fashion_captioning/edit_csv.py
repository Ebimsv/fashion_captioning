import pandas as pd

df = pd.read_csv("annotations/train_title.txt")
# df['id'] = df['id'].astype(str) + ".jpg"
df_2 = df.iloc[:, 0]
df_2.to_csv("annotations/train_title_id.txt", index=False, header=False)

# df_img = df[["id"]]
# df_img.to_csv("df_img.txt", index=False, header=False)

# df_3.to_csv("annotations/df_3.txt", index=False, header=False)

# df1 = pd.read_csv("annotations/1.txt")
# df1.sort_values("id", inplace=True)
# df1.drop_duplicates(keep=False, inplace=True)


# intersection and diff
words1 = set(open("annotations/img.txt").read().split())
words2 = set(open("annotations/train_title_id.txt").read().split())

duplicates = words1.intersection(words2)
uniques = words1.difference(words2).union(words2.difference(words1))

# print("Duplicates(%d):%s" % (len(duplicates), duplicates))
print("\nUniques(%d):%s" % (len(uniques), uniques))
