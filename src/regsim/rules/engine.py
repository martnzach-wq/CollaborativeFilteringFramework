"""Declarative rules (D3.6 gates) evaluated on indicator values (design §8).

A rule is a condition on a demonstrator ``d`` and a target ``t``, written in a small expression language (see
``configs/rules_d36.yaml``). Conditions are evaluated vectorised over all targets. A comparison involving a
missing value is false, so a rule never fires on missing data.

Severity follows the mode: in regsim mode every rule is soft and only adds a flag (decision D9); in D3.6 mode a
hard rule makes the pair ineligible. Rules never change a distance or a score.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path

import numpy as np
import pandas as pd

from regsim.config import KoppenAdjacency, Rule, RuleSet

DEFAULT_RULES = "configs/rules_d36.yaml"
FUNCTIONS = {"max", "min", "abs", "koppen_adjacency", "min_tier_coverage"}
_ALLOWED = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UnaryOp,
    ast.USub,
    ast.Call,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.Tuple,
    ast.Load,
)
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def check_expression(expr: str) -> ast.Expression:
    """Parse ``expr`` and check that it only uses the rule language; raise ``ValueError`` otherwise."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse rule {expr!r}: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"{type(node).__name__} not allowed in rule {expr!r}")
        if isinstance(node, ast.Name) and node.id not in {"d", "t"} | FUNCTIONS:
            raise ValueError(f"unknown name {node.id!r} in rule {expr!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS):
            raise ValueError(f"unknown function in rule {expr!r}")
        if isinstance(node, ast.Attribute) and not (
            isinstance(node.value, ast.Name) and node.value.id in "dt"
        ):
            raise ValueError(f"attributes must be d.<indicator> or t.<indicator> in {expr!r}")
    return tree


def load_rules(path: str | Path = DEFAULT_RULES) -> RuleSet:
    """Read and validate a rule file."""
    return RuleSet.from_yaml(path)


def koppen_adjacency(table: KoppenAdjacency, a, b) -> str | None:
    """``'adjacent'`` or ``'distant'`` for two Köppen-Geiger codes (None if either is missing).

    Pairs with a zone in ``table.observed_zones`` are adjacent only if listed; other pairs are adjacent if they
    share the main group.
    """
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
        return None
    if a == b or tuple(sorted((a, b))) in {tuple(sorted(p)) for p in table.adjacent}:
        return "adjacent"
    if a in table.observed_zones or b in table.observed_zones:
        return "distant"
    return "adjacent" if a[0] == b[0] else "distant"


def _as_values(s: pd.Series) -> pd.Series:
    """Numeric where every present value is numeric, else strings."""
    num = pd.to_numeric(s, errors="coerce")
    return num if num.notna().sum() == s.notna().sum() else s.astype(object).where(s.notna(), None)


class _Evaluator:
    def __init__(self, rules: RuleSet, demo: pd.Series, targets: pd.DataFrame, coverage, prefix: str):
        self.rules, self.demo, self.targets, self.prefix = rules, demo, targets, prefix
        self.index = targets.index
        self.coverage = (
            pd.Series(np.nan, index=self.index)
            if coverage is None
            else pd.Series(coverage).reindex(self.index)
        )
        self._adj = {tuple(sorted(p)) for p in rules.koppen_adjacency.adjacent}

    def column(self, who: str, attr: str) -> pd.Series:
        col = self.prefix + self.rules.aliases.get(attr, attr)
        if who == "t":
            s = self.targets[col] if col in self.targets.columns else pd.Series(np.nan, index=self.index)
            return _as_values(s)
        v = _as_values(pd.Series([self.demo.get(col, np.nan)])).iloc[0]
        return pd.Series(
            [v] * len(self.index), index=self.index, dtype=object if isinstance(v, str) else float
        )

    def eval(self, node):
        method = getattr(self, f"_{type(node).__name__}")
        return method(node)

    def _Expression(self, node):
        return self.eval(node.body)

    def _Constant(self, node):
        return node.value

    def _Tuple(self, node):
        return tuple(self.eval(e) for e in node.elts)

    def _Attribute(self, node):
        return self.column(node.value.id, node.attr)

    def _UnaryOp(self, node):
        return -self.eval(node.operand)

    def _BinOp(self, node):
        a, b = self.eval(node.left), self.eval(node.right)
        with np.errstate(all="ignore"):
            return _BIN[type(node.op)](_float(a), _float(b))

    def _BoolOp(self, node):
        vals = [_bool(self.eval(v), self.index) for v in node.values]
        op = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        return pd.Series(op.reduce([v.to_numpy() for v in vals]), index=self.index)

    def _Compare(self, node):
        left = self.eval(node.left)
        out = pd.Series(True, index=self.index)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = self.eval(comp)
            missing = _missing(left, self.index) | _missing(right, self.index)
            if isinstance(op, ast.In | ast.NotIn):
                hit = pd.Series(left, index=self.index).isin(list(right))
                res = hit if isinstance(op, ast.In) else ~hit
            else:
                res = _CMP[type(op)](left, right)
                res = pd.Series(np.asarray(res, dtype=bool), index=self.index)
            out &= res & ~missing
            left = right
        return out

    def _Call(self, node):
        name = node.func.id
        if name == "min_tier_coverage":
            return self.coverage
        args = [self.eval(a) for a in node.args]
        if name == "koppen_adjacency":
            a, b = (pd.Series(x, index=self.index) for x in args)
            return pd.Series(
                [koppen_adjacency(self.rules.koppen_adjacency, x, y) for x, y in zip(a, b, strict=True)],
                index=self.index,
                dtype=object,
            )
        if name == "abs":
            return np.abs(_float(args[0]))
        f = np.maximum if name == "max" else np.minimum
        out = _float(args[0])
        for a in args[1:]:
            out = f(out, _float(a))
        return out


def _float(x):
    return x.astype(float) if isinstance(x, pd.Series) else float(x)


def _missing(x, index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.isna()
    if isinstance(x, tuple):
        return pd.Series(False, index=index)
    return pd.Series(x is None or (isinstance(x, float) and np.isnan(x)), index=index)


def _bool(x, index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.fillna(False).astype(bool)
    return pd.Series(bool(x), index=index)


def applicable(rules: RuleSet, cic: str | None = None, published_only: bool = False) -> list[Rule]:
    """Rules whose scope contains ``cic`` or ``all`` (all rules if ``cic`` is None)."""
    return [
        r
        for r in rules.rules
        if (cic is None or "all" in r.scope or cic in r.scope) and (r.published or not published_only)
    ]


def evaluate(
    rules: RuleSet,
    demonstrator: pd.Series,
    targets: pd.DataFrame,
    cic: str | None = None,
    coverage: pd.Series | None = None,
    prefix: str = "",
    published_only: bool = False,
) -> pd.DataFrame:
    """Evaluate the rules for one demonstrator against many targets.

    Parameters
    ----------
    rules : RuleSet
        Rules (see :func:`load_rules`).
    demonstrator : Series
        Indicator values of the demonstrator (index: indicator ids).
    targets : DataFrame
        Indicator values of the targets (rows: targets, columns: indicator ids).
    cic : str, optional
        Climate impact context; selects the rules in scope.
    coverage : Series, optional
        Minimum tier coverage per target, used by ``min_tier_coverage``.
    prefix : str
        Prefix of the indicator columns (``"d36_"`` for the observation store).

    Returns
    -------
    DataFrame
        Boolean flag per target (rows) and rule id (columns).
    """
    ev = _Evaluator(rules, demonstrator, targets, coverage, prefix)
    out = {}
    for r in applicable(rules, cic, published_only):
        out[r.id] = _bool(ev.eval(check_expression(r.when)), targets.index)
    return pd.DataFrame(out, index=targets.index)


def gate_summary(
    flags: pd.DataFrame, rules: RuleSet, mode: str = "regsim", severity: str = "published"
) -> pd.DataFrame:
    """Fired rules per target and eligibility.

    ``mode="regsim"``: every rule is soft (decision D9), so all pairs stay eligible. ``mode="d36"``: rules with
    ``severity_<severity>`` = hard make the pair ineligible.
    """
    if mode not in {"regsim", "d36"}:
        raise ValueError(f"unknown mode {mode}")
    sev = {r.id: getattr(r, f"severity_{severity}") for r in rules.rules}
    fired = flags.apply(lambda row: [k for k, v in row.items() if v], axis=1)
    hard = [k for k in flags.columns if sev.get(k) == "hard"] if mode == "d36" else []
    eligible = ~flags[hard].any(axis=1) if hard else pd.Series(True, index=flags.index)
    return pd.DataFrame({"gates_fired": fired.map(", ".join), "eligible": eligible})
