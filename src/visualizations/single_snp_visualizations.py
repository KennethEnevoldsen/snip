"""!pip install seaborn."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    project_path = Path("/home/kce/NLPPred/github/snip")
    read_path = project_path / "data" / "ldak_results"
    save_path = project_path / "docs" / "images" / "single_snp_analysis"

    files = [read_path / f for f in os.listdir(read_path) if f.endswith(".assoc")]

    for i, f in enumerate(files):
        # break
        title = f.stem
        print(f"{title} - {i} / {len(files)}")
        df = pd.read_csv(f, index_col=None, delimiter=" ")
        df = df.reset_index()

        df["-logp"] = -np.log10(df["Wald_P"])

        running_pos = 0
        cumulative_pos = []

        for chrom, group_df in df.groupby("Chromosome"):
            # create a cumulative position for each chromosome
            cumulative_pos.append(group_df["Basepair"] + running_pos)
            running_pos += group_df["Basepair"].max()

        df["cumulative_pos"] = pd.concat(cumulative_pos)
        df["SNP number"] = df.index

        graph = sns.relplot(
            data=df,
            x="SNP number",
            y="-logp",
            aspect=4,
            hue="Chromosome",
            palette="Set1",
            linewidth=0,  # no white border around points
            alpha=0.3,
        )
        threshold = 5e-8
        graph.ax.axhline(y=-np.log10(threshold), color="r", linestyle="--")
        graph.ax.set_title(title)
        graph.ax.set_xlabel("Chromosome position")
        graph.ax.set_ylabel("-log10(p-value)")
        # plot
        save_path.mkdir(parents=True, exist_ok=True)
        graph.savefig(save_path / f"{title}.png")

    for i, f in enumerate(sorted(files)):
        title = f.stem
        print(f"{title} - {i} / {len(files)}")
        df = pd.read_csv(f, index_col=None, delimiter=" ")
        df = df.reset_index()

        n_sig = sum(df["Wald_P"] < 5e-08)
        expected = 5e-08 * len(df)
        print("\tN significant (p < 5x10^-8):", n_sig)
        print("\tExpected N given number of SNPs:", expected)
        print("\tN significant / expected: ", n_sig / expected)
