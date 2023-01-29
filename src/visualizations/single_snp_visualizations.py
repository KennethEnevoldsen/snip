"""!pip install seaborn."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns


def read_assoc(filepath):
    assert str(filepath).endswith(".assoc")
    df = pd.read_csv(str(filepath), index_col=None, delimiter=" ")
    df = df.reset_index()
    return df


def create_single_snp_plot(filepath, save_path, df=None):
    f = Path(filepath)
    title = f.stem
    print("Processing:", title)
    if df is None:
        df = read_assoc(f)

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
    graph.savefig(save_path)


def extract_significant(assoc_df):
    n_sig = sum(df["Wald_P"] < 5e-08)
    expected = 5e-08 * len(df)
    return {
        "N significant (p < 5x10^-8)": n_sig,
        "Expected N given number of SNPs": expected,
        "N significant / expected": n_sig / expected,
    }


if __name__ == "__main__":
    project_path = Path("/home/kce/NLPPred/github/snip")
    read_path = project_path / "data" / "ldak_results"
    save_path = project_path / "docs" / "images" / "single_snp_analysis"
    save_path.mkdir(parents=True, exist_ok=True)

    files = [
        read_path / f
        for f in os.listdir(read_path)
        if f.endswith(".assoc") and f.startswith("chr1")
    ]

    significance_ratios = []
    for i, f in enumerate(sorted(files)):
        title = f.stem
        print(f"Processing: {title} ({i+1}/{len(files)})")
        chrom_range, samples, activation, width = title.split("_")[:4]
        df = read_assoc(f)
        # create_single_snp_plot(f, save_path = save_path / f"{title}.png", df=df)
        sign_rat = extract_significant(df)
        sign_rat = {
            **sign_rat,
            "title": title,
            "chrom_range": chrom_range,
            "samples": samples,
            "activation": activation,
            "width": width,
            "pheno": "height",
        }
        significance_ratios.append(sign_rat)

    # do the same for the new series of phenotypes
    files = list(read_path.glob("chr1-22_20k*/*.assoc"))

    for i, f in enumerate(sorted(files)):
        pheno = f.stem.split(".")[0]
        chrom_range, samples, activation, width = f.parent.stem.split("_")[:4]
        df = read_assoc(f)
        sign_rat = extract_significant(df)
        sign_rat = {
            **sign_rat,
            "title": title,
            "chrom_range": chrom_range,
            "samples": samples,
            "activation": activation,
            "width": width,
            "pheno": pheno,
        }

        significance_ratios.append(sign_rat)
        create_single_snp_plot(f, save_path / f"{pheno}_{title}.png", df=df)

    significance_ratios = pd.DataFrame(significance_ratios)
    significance_ratios.to_csv(
        project_path / "data" / "analysis" / "significance_ratios.csv",
        index=False,
    )
    # select only the columns we want
    report_df = significance_ratios[
        [
            "activation",
            "width",
            "pheno",
            "N significant (p < 5x10^-8)",
            "Expected N given number of SNPs",
            "N significant / expected",
        ]
    ]
    print(report_df.groupby(["pheno", "activation", "width"]).to_markdown(index=False))
