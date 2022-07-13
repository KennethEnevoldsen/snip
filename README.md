<a href="https://github.com/kennethenevoldsen/snip"><img src="https://github.com/KennethEnevoldsen/snip/blob/main/docs/_static/icon.png?raw=true" width="200" align="right" /></a>
# Snip: A utility package handling Single Nucleotide polymorphism data in Python


[![PyPI version](https://badge.fury.io/py/snip.svg)](https://pypi.org/project/snip/)
[![python version](https://img.shields.io/badge/Python-%3E=3.8-blue)](https://github.com/kennethenevoldsen/snip)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/kennethenevoldsen/snip/actions/workflows/pytest.yml/badge.svg)](https://github.com/kennethenevoldsen/snip/actions)
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/c102b02c0430c5e834a7a39abd846130/raw/badge-snip-coverage.json)


A utility package handling Single Nucleotide polymorphism data in Python with the intended use of using in in e.g. PyTorch.

## 🔧 Installation
To get started using this package install it using pip by running the following line in your terminal:

```
pip install git+https://github.com/KennethEnevoldsen/snip
```


For more detailed instructions on installing see the [installation instructions](https://kennethenevoldsen.github.io/snip/installation).


## 👩‍💻 Getting started

To convert `.bed` files to `.zarr`
```
snip convert sample.bed sample.zarr
```

or equivalently:

```
python -m snip convert sample.bed sample.zarr
```

To see a list of possible commands:
```
snip --help
```

To find out more about each command:

```
snip convert --help
```

## 📖 Documentation

| Documentation              |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| 📰 **[News and changelog]** | New additions, changes and version history.                         |
| 🎛 **[API References]**     | The reference for the package API. Including function documentation |
| 🙋 **[FAQ]**                | Frequently asked question                                           |

[usage guides]: https://kennethenevoldsen.github.io/snip/introduction.html
[api references]: https://kennethenevoldsen.github.io/snip/
[News and changelog]: https://kennethenevoldsen.github.io/snip/news.html
[FAQ]: https://kennethenevoldsen.github.io/snip/faq.html

## 💬 Where to ask questions

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |

[github issue tracker]: https://github.com/kennethenevoldsen/snip/issues
[github discussions]: https://github.com/kennethenevoldsen/snip/discussions

