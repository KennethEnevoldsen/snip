<a href="https://github.com/kennethenevoldsen/snip"><img src="https://github.com/KennethEnevoldsen/snip/blob/main/docs/_static/icon.png?raw=true" width="200" align="right" /></a>
# Snip: A package for data handling and model training using Single Nucleotide polymorphism data


[![PyPI version](https://badge.fury.io/py/snip.svg)](https://pypi.org/project/snip/)
[![python version](https://img.shields.io/badge/Python-%3E=3.8-blue)](https://github.com/kennethenevoldsen/snip)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/kennethenevoldsen/snip/actions/workflows/pytest.yml/badge.svg)](https://github.com/kennethenevoldsen/snip/actions)
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/c102b02c0430c5e834a7a39abd846130/raw/badge-snip-coverage.json)


A package for data handling and model training using Single Nucleotide polymorphism data. Implemented in Python and PyTorch.

## 🔧 Installation
To get started using this package install it using pip by running the following line in your terminal:

```
pip install git+https://github.com/KennethEnevoldsen/snip
```


For more detailed instructions on installing see the [installation instructions](https://kennethenevoldsen.github.io/snip/installation).


## Development Setup

To set up the project for development:
```
conda create -n snip python=3.9
conda activate snip
conda install poetry
peotry install
```

## 👩‍💻 Getting started


### Convert
To convert `.bed` files to `.zarr` simply run from your terminal:
```bash
snip convert sample.bed sample.zarr
```

or equivalently:

```bash
python -m snip convert sample.bed sample.zarr
```

### 


### Learn more
To see a list of possible commands:
```bash
snip --help
```

To find out more about each command:

```bash
snip convert --help
```

## Slurm

This project uses slurm.

<br /> 

<details>
    <summary> Slurm quick guide </summary>

**To run a job:**

```bash
sbatch {filename}.sh -A NLPPred
```

Where `A` stands for account and `NLPPred` is the account. 

**Check the status of submitted queue:**
```
squeue -u {username}
```
**See available nodes:**
```
gnodes
```

**SSH to node:**
```bash
ssh {node id}
```

**Run an interactive window:**
```
srun --pty -c 4 --mem=16g bash -A NLPPred
```
Using 4 cores and 16gb memory.

For more on slurm please check out [this site](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands).


</details>

<br /> 


## 💬 Where to ask questions

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |

[github issue tracker]: https://github.com/kennethenevoldsen/snip/issues
[github discussions]: https://github.com/kennethenevoldsen/snip/discussions

