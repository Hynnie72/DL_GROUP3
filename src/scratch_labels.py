import pandas as pd

df = pd.read_csv(r"d:\coursework\CXR8\CXR8\Data_Entry_2017_v2020.csv")
all_labels = set()
for entry in df["Finding Labels"]:
    for label in str(entry).split("|"):
        all_labels.add(label.strip())

disease_labels = sorted(l for l in all_labels if l != "No Finding")
print("Disease labels found:", len(disease_labels))
for i, l in enumerate(disease_labels):
    print(f"  {i}: {repr(l)}")

print()
multi = df[df["Finding Labels"].str.contains("|", regex=False)].head(3)
print(multi[["Image Index","Finding Labels"]].to_string())
