# Contributing

Thanks for your contribution to the pyStoNED!
Please first discuss the change the developers of this repository by creating an issue before making a change.

## Style

For the consistency of the source code, we follow [PEP8 python style guide](https://www.python.org/dev/peps/pep-0008/).

Using `yapf -i` to format the codes:
```bash
$ yapf -i pystoned/example.py
```

Using `pylint` to check the other requirements:
```bash
$ pylint pystoned/example.py
```

## Commit message
Example:
```
refactor(pyStoNED): Rename the variables in biMat.py

For fitting the requirement of PEP8 style, 
I renamed the variables in biMat.py:
* ... to ...
* ... to ...
* ... to ...

Fixes #0000
```

Please write 1 commit message for one modification. Do not contain multiple changes in 1 commit. If the commit fixes any issue, please add `Fixes #xxxx` to the end of the commit message.

The first line should not be longer than 70 characters, the second line is always blank and other lines should be wrapped at 80 characters. The type should always be lowercase as shown below:

* feat: (new feature for the user, not a new feature for build script)
* fix: (bug fix for the user, not a fix to a build script)
* docs: (changes to the documentation)
* style: (formatting, missing semi colons, etc; no production code change)
* refactor: (refactoring code, eg. renaming a variable)
* test: (adding tests, refactoring tests; no production code change)

Reference: 
* https://seesparkbox.com/foundry/semantic_commit_messages
* http://karma-runner.github.io/1.0/dev/git-commit-msg.html

## Pull Request Process

1. Create a Fork of the project to your own repository.
2. Create a Branch in your Fork.
3. Make your contributions.
4. Make a Test that checks the change you did.
5. Format your code by `yapf` to `PEP8` style.
6. Create a Pull Request (PR) and Provide your test in it.
7. We will review your PR and discuss with you.

Thanks for your contributions!