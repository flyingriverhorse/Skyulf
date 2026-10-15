"""Heuristic static lint for batch-level code in apply-time project callbacks.

This is a heuristic, not a proof. It parses the saved source of a step's
apply/rule functions and the same-module helpers they call, and flags calls
that read a whole batch (column statistics, ranking, shifting, grouping,
sorting, ``len(df)``, ``df.shape``) on values that come from the input frame.
``learn()`` may use batch statistics: learning on training rows is its job.
It also flags writes to module globals anywhere in the step's reachable
functions, because each Spark worker process keeps its own globals.

Only same-module helpers called by plain name are followed; other modules,
dynamic attribute access and aliases built in other ways are not seen. The
dynamic probe still runs afterwards.
"""

import ast
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LINT_VERSION = 1

BATCH_METHODS = frozenset(
    {
        "mean",
        "median",
        "quantile",
        "std",
        "var",
        "sem",
        "min",
        "max",
        "sum",
        "prod",
        "any",
        "all",
        "count",
        "idxmin",
        "idxmax",
        "agg",
        "aggregate",
        "describe",
        "value_counts",
        "unique",
        "nunique",
        "mode",
        "rank",
        "shift",
        "diff",
        "pct_change",
        "rolling",
        "expanding",
        "ewm",
        "groupby",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "duplicated",
        "drop_duplicates",
        "sort_values",
        "sort_index",
        "nlargest",
        "nsmallest",
    }
)
NUMPY_REDUCTIONS = frozenset(
    {
        "mean",
        "median",
        "percentile",
        "quantile",
        "std",
        "var",
        "average",
        "nanmean",
        "nanmedian",
        "nanpercentile",
        "nanstd",
        "nanvar",
    }
)
_MUTATORS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "update",
        "setdefault",
        "pop",
        "popitem",
        "clear",
        "remove",
        "discard",
        "add",
        "sort",
        "reverse",
    }
)
# Reductions that have a per-row form with axis=1; the other batch methods have no such fix.
AXIS_REDUCTIONS = frozenset(
    {
        "mean",
        "median",
        "quantile",
        "std",
        "var",
        "sem",
        "min",
        "max",
        "sum",
        "prod",
        "any",
        "all",
        "count",
        "idxmin",
        "idxmax",
        "nunique",
        "agg",
        "aggregate",
    }
)
_ACCESSORS = frozenset({"str", "dt", "cat"})
_NUMPY = frozenset({"np", "numpy"})
_NAMESPACES = frozenset({"globals", "vars"})
STATISTIC = "reads a batch statistic"
OTHER_ROWS = "depends on other rows"
BATCH_SIZE = "reads the batch size"
MODULE_STATE = "writes module state"


@dataclass(frozen=True)
class LintFinding:
    """One flagged expression with its saved source location."""

    file: str
    line: int
    call: str
    problem: str


@dataclass(frozen=True)
class _Module:
    """Parsed saved source of one module and its top-level definitions."""

    text: str
    display: str
    tree: ast.Module
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    globals: frozenset[str]


def _module_text(code_file: str) -> tuple[str, str]:
    """Read saved project source, or a real file during local preview."""
    if code_file.startswith("<") and code_file.endswith(">"):
        inner = code_file[1:-1]
        root, separator, relative = inner.partition("/")
        if separator:
            from .project_package import _FINDER  # noqa: PLC0415

            return _FINDER.packages[root][relative], relative
        return vars(sys.modules[inner])["__skyulf_source__"], "preprocessing.py"
    path = Path(code_file)
    return path.read_text(encoding="utf-8"), path.name


def _parse(code_file: str) -> _Module:
    """Parse once and index top-level functions and module-level names."""
    try:
        text, display = _module_text(code_file)
    except (KeyError, OSError) as exc:
        raise ValueError(f"Cannot read the saved source of {code_file} for the lint.") from exc
    tree = ast.parse(text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    return _Module(text, display, tree, functions, frozenset(_module_names(tree)))


def _module_names(tree: ast.Module) -> set[str]:
    """Name every module-level binding: values, classes, functions and imports.

    Writing an attribute of any of them (a class or function attribute, or an
    imported module's state such as os.environ) changes state that a fresh
    worker process lacks.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= _target_names(list(node.targets))
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            names |= _target_names([node.target])
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(alias.asname or alias.name).partition(".")[0] for alias in node.names}
    return names


def _roots(node: ast.AST | None) -> set[str]:
    """Collect names an expression is computed from, through calls and operators."""
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    found: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, ast.Lambda | ast.comprehension | ast.keyword):
            found |= _roots(child)
        elif isinstance(child, ast.keyword):
            found |= _roots(child.value)
    return found


def _params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return positional and keyword parameter names in declaration order."""
    arguments = node.args
    return [arg.arg for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]


def _derived(node: ast.FunctionDef | ast.AsyncFunctionDef, seeds: set[str]) -> set[str]:
    """Follow simple assignments from frame parameters to local aliases."""
    derived = set(seeds)
    assignments = [item for item in ast.walk(node) if isinstance(item, ast.Assign | ast.AnnAssign)]
    changed = True
    while changed:
        changed = False
        for item in assignments:
            if not _roots(item.value) & derived:
                continue
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for name in _target_names(targets):
                if name not in derived:
                    derived.add(name)
                    changed = True
    return derived


def _target_names(targets: list[ast.expr]) -> set[str]:
    """Return plain names bound by assignment targets, including tuple unpacking."""
    names = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple | ast.List):
            names |= _target_names(list(target.elts))
    return names


def _row_axis(call: ast.Call, position: int = 0) -> bool:
    """Accept a per-row axis: axis=1, axis='columns' or a positional 1."""
    values = [keyword.value for keyword in call.keywords if keyword.arg == "axis"]
    if len(call.args) > position:
        values.append(call.args[position])
    return any(
        isinstance(value, ast.Constant) and value.value in (1, "columns") for value in values
    )


def _batch_call(call: ast.Call, derived: set[str]) -> str | None:
    """Classify one call as a batch statistic or batch-size read on frame values."""
    func = call.func
    if isinstance(func, ast.Name) and func.id == "len" and call.args:
        return BATCH_SIZE if _roots(call.args[0]) & derived else None
    if not isinstance(func, ast.Attribute):
        return None
    return _method_statistic(func, call, derived)


def _method_statistic(func: ast.Attribute, call: ast.Call, derived: set[str]) -> str | None:
    """Flag numpy reductions over frame values and batch methods on frame values.

    String, datetime and category accessor methods work row by row.
    """
    receiver = func.value
    if isinstance(receiver, ast.Name) and receiver.id in _NUMPY:
        return _numpy_statistic(func, call, derived)
    if isinstance(receiver, ast.Attribute) and receiver.attr in _ACCESSORS:
        return None
    if func.attr not in BATCH_METHODS or not _roots(receiver) & derived or _row_axis(call):
        return None
    return STATISTIC if func.attr in AXIS_REDUCTIONS else OTHER_ROWS


def _numpy_statistic(func: ast.Attribute, call: ast.Call, derived: set[str]) -> str | None:
    """Flag np.mean(column) and similar reductions unless they use axis 1."""
    frames = any(_roots(arg) & derived for arg in call.args[:1])
    reduction = func.attr in NUMPY_REDUCTIONS and frames and not _row_axis(call, 1)
    return STATISTIC if reduction else None


class _Linter:
    """Walk reachable callbacks and helpers, collecting deduplicated findings."""

    def __init__(self) -> None:
        self.modules: dict[str, _Module] = {}
        self.findings: dict[tuple[str, int, str], LintFinding] = {}
        self.seen: set[tuple[str, str, frozenset[str], bool]] = set()

    def module(self, code_file: str) -> _Module:
        """Parse each saved module once."""
        if code_file not in self.modules:
            self.modules[code_file] = _parse(code_file)
        return self.modules[code_file]

    def add(self, module: _Module, node: ast.AST, problem: str) -> None:
        """Record one finding at the node's saved source location."""
        call = ast.get_source_segment(module.text, node) or type(node).__name__
        line = getattr(node, "lineno", 0)
        self.findings[(module.display, line, call)] = LintFinding(
            module.display, line, call, problem
        )

    def visit(self, code_file: str, name: str, seeds: frozenset[str], apply: bool) -> None:
        """Lint one top-level function, then the same-module helpers it calls."""
        key = (code_file, name, seeds, apply)
        module = self.module(code_file)
        node = module.functions.get(name)
        if key in self.seen or node is None:
            return
        self.seen.add(key)
        local = _locals(node)
        derived = _derived(node, set(seeds)) if apply else set()
        writable = frozenset((module.globals - local) | _shared_locals(node))
        exempt = _guard_nodes(node)
        for item in ast.walk(node):
            self.check(module, item, derived, writable, apply=apply and id(item) not in exempt)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                self.follow(code_file, module, item, derived, local, apply=apply)

    def check(
        self,
        module: _Module,
        item: ast.AST,
        derived: set[str],
        writable: frozenset[str],
        *,
        apply: bool,
    ) -> None:
        """Flag shared-state writes everywhere and batch reads on apply paths."""
        write = _module_write(item, writable)
        if write is not None:
            self.add(module, write, MODULE_STATE)
        if not apply:
            return
        if isinstance(item, ast.Call):
            problem = _batch_call(item, derived)
            if problem is not None:
                self.add(module, item, problem)
        elif isinstance(item, ast.Attribute) and item.attr == "shape":
            if _roots(item.value) & derived:
                self.add(module, item, BATCH_SIZE)

    def follow(
        self,
        code_file: str,
        module: _Module,
        call: ast.Call,
        derived: set[str],
        local: set[str],
        *,
        apply: bool,
    ) -> None:
        """Lint a called same-module helper with the parameters that receive frame values."""
        name = call.func.id if isinstance(call.func, ast.Name) else ""
        helper = module.functions.get(name)
        if helper is None or name in local:
            return
        params = _params(helper)
        seeds = {
            params[i]
            for i, arg in enumerate(call.args)
            if i < len(params) and _roots(arg) & derived
        }
        seeds |= {kw.arg for kw in call.keywords if kw.arg in params and _roots(kw.value) & derived}
        self.visit(code_file, name, frozenset(seeds), apply)


def _locals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return names bound inside a function unless it declares them global."""
    declared = {
        name for item in ast.walk(node) if isinstance(item, ast.Global) for name in item.names
    }
    bound = set(_params(node))
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            bound.add(item.id)
        elif isinstance(item, ast.arg):
            bound.add(item.arg)
    return bound - declared


def _shared_locals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return local names that alias shared state: mutable defaults and globals()."""
    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    tail = positional[len(positional) - len(arguments.defaults) :]
    pairs = list(zip(tail, arguments.defaults, strict=True))
    pairs += [
        (arg, d) for arg, d in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True) if d
    ]
    shared = {arg.arg for arg, default in pairs if _mutable(default)}
    for item in ast.walk(node):
        if isinstance(item, ast.Assign) and _namespace_call(item.value):
            shared |= _target_names(list(item.targets))
    return shared


def _mutable(node: ast.expr) -> bool:
    """Recognize a mutable default value, which persists across calls in one process."""
    if isinstance(node, ast.List | ast.Dict | ast.Set):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "dict", "set", "defaultdict", "OrderedDict"}
    )


def _namespace_call(node: ast.AST) -> bool:
    """Recognize globals() and vars() calls, whose result is the module namespace."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _NAMESPACES
        and not node.args
    )


def _guard_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    """Return ids of nodes that only guard input, so their batch reads are fine.

    That covers a validation if-test whose body only raises, an empty-batch
    check (len(df) == 0, df.empty) whose body only returns or raises, and the
    column count df.shape[1].
    """
    exempt: set[int] = set()
    for item in ast.walk(node):
        guard = _input_guard(item)
        if guard is not None:
            exempt |= {id(child) for child in ast.walk(guard)}
        elif _repeated_constant(item):
            exempt |= {id(child) for child in ast.walk(item)}
        else:
            shape = _column_count(item)
            if shape is not None:
                exempt.add(id(shape))
    return exempt


def _input_guard(item: ast.AST) -> ast.expr | None:
    """Return the test of an if that only raises, or only returns early when empty."""
    if not isinstance(item, ast.If) or len(item.body) != 1:
        return None
    if isinstance(item.body[0], ast.Raise):
        return item.test
    returns = isinstance(item.body[0], ast.Return) and _empty_check(item.test)
    return item.test if returns else None


def _repeated_constant(item: ast.AST) -> bool:
    """Recognize [value] * len(df): one constant per row, independent of other rows."""
    if not isinstance(item, ast.BinOp) or not isinstance(item.op, ast.Mult):
        return False
    sides = (item.left, item.right)
    single = any(isinstance(side, ast.List | ast.Tuple) and len(side.elts) == 1 for side in sides)
    counted = any(
        isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "len"
        for side in sides
    )
    return single and counted


def _column_count(item: ast.AST) -> ast.Attribute | None:
    """Return the shape node of df.shape[1], the column count, which ignores batch rows."""
    if not isinstance(item, ast.Subscript) or not isinstance(item.value, ast.Attribute):
        return None
    column = isinstance(item.slice, ast.Constant) and item.slice.value == 1
    return item.value if item.value.attr == "shape" and column else None


def _empty_check(test: ast.expr) -> bool:
    """Recognize len(df) == 0, not len(df) and df.empty tests."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test = test.operand
    if isinstance(test, ast.Attribute):
        return test.attr == "empty"
    if isinstance(test, ast.Compare):
        sides = [test.left, *test.comparators]
        return any(isinstance(side, ast.Constant) and side.value in (0, 1) for side in sides)
    return isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "len"


def _base(node: ast.expr) -> str | None:
    """Return the name an attribute/subscript chain writes into, ignoring index values."""
    while isinstance(node, ast.Subscript | ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _module_write(item: ast.AST, module_names: frozenset[str]) -> ast.AST | None:
    """Return the node that writes module state, if this statement or call does."""
    if isinstance(item, ast.Global):
        return item
    if isinstance(item, ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete):
        targets = item.targets if isinstance(item, ast.Assign | ast.Delete) else [item.target]
        return next((target for target in targets if _writes(target, module_names)), None)
    return item if isinstance(item, ast.Call) and _mutates(item, module_names) else None


def _mutates(call: ast.Call, module_names: frozenset[str]) -> bool:
    """Report whether a call such as CACHE.update(...) mutates shared state."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in _MUTATORS:
        return False
    inner = func.value
    while isinstance(inner, ast.Subscript | ast.Attribute):
        inner = inner.value
    return _namespace_call(inner) or _base(func.value) in module_names


def _writes(target: ast.expr, module_names: frozenset[str]) -> bool:
    """Report whether an item or attribute assignment writes into shared state."""
    if not isinstance(target, ast.Subscript | ast.Attribute):
        return False
    inner = target
    while isinstance(inner, ast.Subscript | ast.Attribute):
        inner = inner.value
    return _namespace_call(inner) or _base(target) in module_names


def _entry(function: Callable[..., Any]) -> tuple[str, str]:
    """Locate a callback's saved module and its top-level name."""
    code = getattr(function, "__code__", None)
    if code is None:
        raise ValueError(f"Cannot lint {function!r}: it is not a plain Python function.")
    return code.co_filename, getattr(function, "__name__", "")


def lint_callbacks(
    apply: Sequence[Callable[..., Any]],
    learn: Sequence[Callable[..., Any]] = (),
    *,
    frames: int = 1,
) -> tuple[LintFinding, ...]:
    """Return findings for apply-time callbacks and module-state writes in learn().

    ``frames`` is how many leading parameters of each apply callback receive
    frame data (2 for scoring output rules, which also receive predictions).
    """
    linter = _Linter()
    for function in apply:
        code_file, name = _entry(function)
        node = linter.module(code_file).functions.get(name)
        seeds = frozenset(_params(node)[:frames]) if node is not None else frozenset()
        linter.visit(code_file, name, seeds, True)
    for function in learn:
        code_file, name = _entry(function)
        linter.visit(code_file, name, frozenset(), False)
    return tuple(sorted(linter.findings.values(), key=lambda item: (item.file, item.line)))
