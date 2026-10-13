"""Command-line interface."""

from __future__ import annotations

import typer

app = typer.Typer(help="regsim: similarity and prediction for multi-level regional indicators")


@app.command()
def build(config: str):
    """Fetch, harmonise and store observations."""
    raise NotImplementedError("M1")


@app.command()
def fit(config: str):
    raise NotImplementedError("M3")


@app.command()
def evaluate(config: str):
    raise NotImplementedError("M2")


@app.command()
def similar(config: str, geo_id: str, k: int = 10, lens: str = "all"):
    raise NotImplementedError("M4")


@app.command()
def typology(config: str, lens: str = "all", level: int = 3):
    raise NotImplementedError("M5")


@app.command()
def predict(config: str, geo_id: str):
    raise NotImplementedError("M3")


@app.command()
def report(config: str):
    raise NotImplementedError("M6")
