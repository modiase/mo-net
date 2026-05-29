"""Entrypoint for `python -m mo_net.samples.word2vec`.

Real CLI lives in :mod:`mo_net.samples.word2vec.cli` — importing the
package registers the train / inspection commands onto the click group.
"""

from mo_net.samples.word2vec.cli import cli

if __name__ == "__main__":
    cli()
