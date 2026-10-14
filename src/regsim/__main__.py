"""`python -m regsim ...`: the command-line interface without the `regsim` script on the PATH."""

from regsim.api.cli import app

app(prog_name="regsim")
