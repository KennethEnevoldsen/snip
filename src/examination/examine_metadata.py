import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def get_metadata_paths():
    path = Path("/home/kce/NLPPred/github/snip/data/compressed/whole_geno")

    paths = []
    for subfolder in os.listdir(path):
        content = os.listdir(path / subfolder)
        if "metadata.jsonl" in content:
            paths.append(path / subfolder / "metadata.jsonl")
    return paths


def load_metadata_as_df():
    paths = get_metadata_paths()
    dfs = []
    for path in paths:
        df = pd.read_json(path, lines=True)
        df["path"] = str(path).split("/")[-2]
        dfs.append(df)
    return pd.concat(dfs)


df = load_metadata_as_df()
df

df["converged"].unique()
df.groupby(["path"])["mean individual correlation (training set)"].mean()

fig = df.hist(
    column="mean individual correlation (training set)",
    by="path",
    figsize=(10, 9),
)
[x.title.set_size(6) for x in fig.ravel()]
plt.show()

fig = df.hist(column="mean snp correlation (traning set)", by="path", figsize=(10, 9))
[x.title.set_size(6) for x in fig.ravel()]
plt.show()

fig = df.hist(column="n. trivial snps", by="path", figsize=(10, 9))
[x.title.set_size(6) for x in fig.ravel()]
plt.show()


fig = df.hist(column="n. trivial snps", by="path", figsize=(10, 9))
[x.title.set_size(6) for x in fig.ravel()]
plt.show()


fig = df.hist(column="n. trivial snps", by="n", figsize=(10, 9))
[x.title.set_size(6) for x in fig.ravel()]
plt.show()

df_chr1_200k = df[df["path"] == "chr1_200k_relu_512silvery-rain-169_2023-01-16"]
(df_chr1_200k["n. trivial snps"] < 200).sum()
