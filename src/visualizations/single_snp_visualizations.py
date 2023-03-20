"""!pip install seaborn."""
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns


def read_assoc(filepath):
    assert str(filepath).endswith(".assoc")
    df = pd.read_csv(str(filepath), index_col=None, delimiter=" ")
    df = df.reset_index()
    return df


def get_reml_path(filepath):
    f = Path(filepath)
    stem_no_quant = ".".join(f.stem.split(".")[:-1])
    reml_output = f"{stem_no_quant}.reml1.reml"
    reml_path = f.parent / reml_output
    assert reml_path.exists(), f"REML file {reml_path} does not exist"
    return reml_path


def get_heritability(filepath):
    """Get the heritability estimates for the phenotypes."""
    reml_path = get_reml_path(filepath)
    with open(reml_path, "r") as f:
        lines = f.readlines()
    _, her_all, her_sd = lines[-1].split(" ")[:3]
    return float(her_all), float(her_sd)


def create_single_snp_plot(filepath, save_path, df=None):
    # check if savepath already exists
    if save_path.exists():
        print("Skipping:", save_path)
        return
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


def get_info_from_filepath(filepath):
    f = Path(filepath)
    if "uncompressed" in f.stem or "prune" in f.stem:
        pheno, _, compression = f.stem.split(".")[:3]
        title = f.stem
        chrom_range = "chr1-22"
        samples = "20k"
        width = ""
        activation = ""
        if compression == "prune":
            compression = f"Pruning (r2={'.'.join(f.stem.split('.')[3:-1])})"
    elif f.parent.stem.startswith("chr"):
        pheno = f.stem.split(".")[0]
        title = f.parent.stem
        chrom_range, samples, activation, width = title.split("_")[:4]
        if "compression" in f.parent.stem:
            compression_factor = f.parent.name.split("_")[-4]
        else:
            compression_factor = 2
        if compression_factor == "identity":
            print("test")
        compression = f"Autoencoder x{compression_factor}"
    return title, pheno, chrom_range, samples, activation, width, compression


def create_table_row(filepath, df):
    (
        title,
        pheno,
        chrom_range,
        samples,
        activation,
        width,
        compression,
    ) = get_info_from_filepath(filepath)

    sign_rat = extract_significant(df)
    her, her_sd = get_heritability(f)

    return {
        **sign_rat,
        "title": title,
        "chrom_range": chrom_range,
        "samples": samples,
        "activation": activation,
        "width": width,
        "pheno": pheno,
        "heritability": her,
        "heritability_sd": her_sd,
        "compression": compression,
    }


if __name__ == "__main__":
    project_path = Path("/home/kce/NLPPred/github/snip")
    read_path = project_path / "data" / "ldak_results"
    save_path = project_path / "docs" / "images" / "single_snp_analysis"
    save_path.mkdir(parents=True, exist_ok=True)

    files = list(read_path.glob("chr1-22_20k*/*.assoc"))
    # add uncompressed
    # files += list(read_path.glob("uncompressed/*.assoc"))
    # files += list(read_path.glob("pruning/*.assoc"))

    table = []
    for i, f in enumerate(sorted(files)):
        df = read_assoc(f)
        row = create_table_row(f, df)
        print(f"Processing {i+1}/{len(files)}: {f.parent.name}/{f.stem}")
        table.append(row)
        create_single_snp_plot(
            f,
            save_path / f"{row['pheno']}_{row['title']}.png",
            df=df,
        )

    results_df = pd.DataFrame(table)
    results_df.to_csv(
        project_path / "data" / "analysis" / "significance_ratio_and_heritability.csv",
        index=False,
    )
    # select only the columns we want for the report
    report_df = results_df[
        [
            "compression",
            "activation",
            "width",
            "pheno",
            "N significant (p < 5x10^-8)",
            "Expected N given number of SNPs",
            "N significant / expected",
            "heritability",
        ]
    ]
    print(report_df.to_markdown(index=False))

    report_df
