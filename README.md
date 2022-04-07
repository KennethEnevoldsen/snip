# Name, the python package template which makes your dream come true
This is a GitHub template for python packages.


## Places to change thing:

### Setup
- [ ] setup.py: update `name`
- [ ] setup.cfg: `name`, `description`, `install_requires`, `url`
- [ ] about.py: `__download_url__`, `__title__`
- [ ] Name of folder `src/name` to package name

### Workflows:
- [ ] pytest: 
  - [ ] `name` in line 35 and 54
  - [ ] Add secret `GIST_DOCS_SCOPE` til GitHub action. This is a GitHub authentication token for `repo, read:repo hook, gist`.
  - [ ] Add a gist and gist ID for the coverage comment
- [ ] publish_to_pypi 
  - [ ] `name` in line 35 and 54
  - [ ] Add secret: `PYPI_API_TOKEN` to GitHub secrets
- [ ] Delete `.github/workflows/tutorials.yml` if you do not use tutorials

### Documentation
- [ ] Change project name in line 33 in `docs/conf.py`
- [ ] Change citation in `docs/faq.rst`
- [ ] `name` in line 1, 4, 5, 23 and description in line 7 in `docs/index.rst`
- [ ] `name` in line 7 in `docs/installation.rst`
- [ ] Update `docs/news.rst`
- [ ] Fill out `README_outline.md`
  - [ ] `name` in line 1, 2, 5, 6, 8, 9, 11 ...
  - [ ] `replace` the gist ID in line 10 for the coverage comment
- [ ] replace `README.md` with `README_outline.md`
- [ ] Replace icong in `docs/_static`, create a favicon using [favicon.io](https://favicon.io/favicon-converter/)
