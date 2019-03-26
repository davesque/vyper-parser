# vyper-parser

[![Build Status](https://circleci.com/gh/davesque/vyper-parser.svg?style=shield)](https://circleci.com/gh/davesque/vyper-parser)

An experimental parser for [vyper](https://github.com/ethereum/vyper).

## Developer Setup

If you would like to hack on vyper-parser, please check out the [Snake Charmers
Tactical Manual](https://github.com/ethereum/snake-charmers-tactical-manual)
for information on how we do:

- Testing
- Pull Requests
- Code Style
- Documentation

### Development Environment Setup

You can set up your dev environment with:

```sh
git clone git@github.com:davesque/vyper-parser.git
cd vyper-parser
virtualenv -p python3 venv
. venv/bin/activate
pip install -e .[dev]
```

### Testing Setup

During development, you might like to have tests run on every file save.

Show flake8 errors on file change:

```sh
# Test flake8
when-changed -v -s -r -1 vyper_parser/ tests/ -c "clear; flake8 vyper_parser tests && echo 'flake8 success' || echo 'error'"
```

Run multi-process tests in one command, but without color:

```sh
# in the project root:
pytest --numprocesses=4 --looponfail --maxfail=1
# the same thing, succinctly:
pytest -n 4 -f --maxfail=1
```

Run in one thread, with color and desktop notifications:

```sh
cd venv
ptw --onfail "notify-send -t 5000 'Test failure ⚠⚠⚠⚠⚠' 'python 3 test on vyper-parser failed'" ../tests ../vyper_parser
```
