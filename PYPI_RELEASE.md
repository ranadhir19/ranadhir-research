# PyPI Release Checklist — ghosh-research

Do NOT run these until the repo is final and reviewed.

## 1. Prepare
- [ ] `python3 -m pip install --upgrade build twine` (use `--break-system-packages` only if needed, or a venv)
- [ ] Bump `version` in `pyproject.toml` if this is not the first release
- [ ] `git tag v0.1.0 && git push origin v0.1.0`

## 2. Build
```bash
cd ~/workspace/research
python3 -m build
```
Expected: `dist/ghosh_research-0.1.0-py3-none-any.whl` and `dist/ghosh_research-0.1.0.tar.gz`.

## 3. Check
```bash
python3 -m twine check dist/*
```
Must report no errors.

## 4. Upload (needs a PyPI API token)
1. Create an account at https://pypi.org/account/register/
2. Generate an API token at https://pypi.org/manage/account/token/ (scope: project `ghosh-research` after first upload, or entire account)
3. Upload:
```bash
python3 -m twine upload dist/*
# username: __token__
# password: <paste the token>
```

## 5. Verify
- [ ] https://pypi.org/project/ghosh-research/ shows the new version
- [ ] Fresh install works: `pip install ghosh-research` in a clean venv, then `python -c "import ghosh_research; print(ghosh_research.__version__)"`
