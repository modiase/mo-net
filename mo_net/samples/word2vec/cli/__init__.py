"""Click CLI for the word2vec sample.

The :data:`cli` group is the entry point — :mod:`mo_net.samples.word2vec.__main__`
calls it. Subcommands register on import: importing the train command
attaches it to ``cli``, same for the inspection commands.
"""

from mo_net.samples.word2vec.cli.group import cli

# Importing for side effects: each module decorates its command onto `cli`.
from mo_net.samples.word2vec.cli import inspect, train  # noqa: F401

__all__ = ["cli"]
