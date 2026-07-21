"""Calculateur mathematique local, borne et sans execution arbitraire.

Le moteur analyse une expression avec :mod:`ast` en mode ``eval`` puis
interprete lui-meme un sous-ensemble numerique explicite. Il n'utilise jamais
``eval``/``exec`` et n'a aucune dependance envers la base de memoire.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
import statistics
import time
from typing import Any, Final
import unicodedata


REGISTRY_VERSION: Final = "math-core-v1"
_MAX_JSON_SAFE_INTEGER: Final = 9_007_199_254_740_991


class MathEngineError(ValueError):
    """Erreur publique bornee pouvant etre retournee telle quelle par l'API."""

    def __init__(self, message: str, *, code: str = "math_error"):
        clean_message = " ".join(str(message).split())[:500]
        clean_code = "".join(
            character
            for character in str(code).strip().lower()
            if character.isalnum() or character in "_-"
        )[:80]
        self.message = clean_message or "Expression mathematique invalide."
        self.code = clean_code or "math_error"
        super().__init__(self.message)


@dataclass(frozen=True, slots=True)
class MathLimits:
    """Quotas appliques avant et pendant chaque calcul."""

    max_expression_chars: int = 1_024
    max_nodes: int = 256
    max_depth: int = 24
    max_collection_items: int = 128
    max_function_args: int = 128
    max_integer_bits: int = 4_096
    max_output_digits: int = 1_200
    max_exponent: int = 1_000
    max_factorial: int = 500
    max_combinatorial_n: int = 1_000
    max_round_digits: int = 100

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} doit etre un entier strictement positif")


@dataclass(frozen=True, slots=True)
class _FunctionSpec:
    name: str
    category: str
    signature: str
    description: str
    exact: str
    min_args: int
    max_args: int

    def public(self) -> dict[str, Any]:
        exact_flag = (
            True
            if self.exact == "true"
            else False
            if self.exact == "false"
            else None
        )
        return {
            "name": self.name,
            "category": self.category,
            "signature": self.signature,
            "description": self.description,
            "exact": exact_flag,
            "exactness": self.exact,
            "learning_level": 4,
            "maturity": "operational",
            "domain": "arguments numeriques reels et finis",
            "returns": "entier, decimal fini ou fraction exacte",
        }


_CATEGORY_INFO: Final = {
    "arithmetic": ("Arithmetique", "Operations, arrondis et reductions numeriques."),
    "roots_logs": ("Racines et logarithmes", "Racines, exponentielles et logarithmes."),
    "trigonometry": ("Trigonometrie", "Angles et fonctions trigonometriques reelles."),
    "integers_combinatorics": (
        "Entiers et combinatoire",
        "Algorithmes entiers avec bornes explicites.",
    ),
    "statistics": ("Statistiques", "Statistiques descriptives sur series bornees."),
    "exact": ("Calcul exact", "Construction de fractions rationnelles exactes."),
}


def _spec(
    name: str,
    category: str,
    signature: str,
    description: str,
    *,
    exact: str = "depends",
    min_args: int = 1,
    max_args: int = 1,
) -> _FunctionSpec:
    return _FunctionSpec(
        name, category, signature, description, exact, min_args, max_args
    )


_SPECS: Final = (
    _spec("abs", "arithmetic", "abs(x)", "Valeur absolue.", exact="preserved"),
    _spec("round", "arithmetic", "round(x[, digits])", "Arrondi decimal borne.", min_args=1, max_args=2),
    _spec("floor", "arithmetic", "floor(x)", "Entier inferieur ou egal."),
    _spec("ceil", "arithmetic", "ceil(x)", "Entier superieur ou egal."),
    _spec("trunc", "arithmetic", "trunc(x)", "Troncature vers zero."),
    _spec("sum", "arithmetic", "sum(sequence)", "Somme d'une serie bornee.", exact="preserved"),
    _spec("fsum", "arithmetic", "fsum(sequence)", "Somme flottante precise.", exact="false"),
    _spec("prod", "arithmetic", "prod(sequence)", "Produit d'une serie bornee.", exact="preserved"),
    _spec("min", "arithmetic", "min(sequence)", "Minimum d'une serie bornee.", exact="preserved"),
    _spec("max", "arithmetic", "max(sequence)", "Maximum d'une serie bornee.", exact="preserved"),
    _spec("sqrt", "roots_logs", "sqrt(x)", "Racine carree reelle.", exact="false"),
    _spec("isqrt", "roots_logs", "isqrt(n)", "Racine carree entiere.", exact="true"),
    _spec("exp", "roots_logs", "exp(x)", "Exponentielle naturelle.", exact="false"),
    _spec("log", "roots_logs", "log(x[, base])", "Logarithme naturel ou dans une base.", exact="false", min_args=1, max_args=2),
    _spec("log2", "roots_logs", "log2(x)", "Logarithme en base deux.", exact="false"),
    _spec("log10", "roots_logs", "log10(x)", "Logarithme en base dix.", exact="false"),
    _spec("hypot", "roots_logs", "hypot(x, ...)", "Norme euclidienne stable.", exact="false", min_args=1, max_args=16),
    _spec("sin", "trigonometry", "sin(x)", "Sinus en radians.", exact="false"),
    _spec("cos", "trigonometry", "cos(x)", "Cosinus en radians.", exact="false"),
    _spec("tan", "trigonometry", "tan(x)", "Tangente en radians.", exact="false"),
    _spec("asin", "trigonometry", "asin(x)", "Arc sinus reel.", exact="false"),
    _spec("acos", "trigonometry", "acos(x)", "Arc cosinus reel.", exact="false"),
    _spec("atan", "trigonometry", "atan(x)", "Arc tangente.", exact="false"),
    _spec("atan2", "trigonometry", "atan2(y, x)", "Angle oriente de deux coordonnees.", exact="false", min_args=2, max_args=2),
    _spec("degrees", "trigonometry", "degrees(x)", "Radians vers degres.", exact="false"),
    _spec("radians", "trigonometry", "radians(x)", "Degres vers radians.", exact="false"),
    _spec("gcd", "integers_combinatorics", "gcd(n, ...)", "Plus grand commun diviseur.", exact="true", min_args=1, max_args=16),
    _spec("lcm", "integers_combinatorics", "lcm(n, ...)", "Plus petit commun multiple.", exact="true", min_args=1, max_args=16),
    _spec("factorial", "integers_combinatorics", "factorial(n)", "Factorielle entiere bornee.", exact="true"),
    _spec("comb", "integers_combinatorics", "comb(n, k)", "Nombre de combinaisons.", exact="true", min_args=2, max_args=2),
    _spec("perm", "integers_combinatorics", "perm(n[, k])", "Nombre de permutations.", exact="true", min_args=1, max_args=2),
    _spec("mean", "statistics", "mean(sequence)", "Moyenne arithmetique.", exact="preserved"),
    _spec("fmean", "statistics", "fmean(sequence)", "Moyenne flottante.", exact="false"),
    _spec("median", "statistics", "median(sequence)", "Mediane.", exact="depends"),
    _spec("median_low", "statistics", "median_low(sequence)", "Mediane basse.", exact="preserved"),
    _spec("median_high", "statistics", "median_high(sequence)", "Mediane haute.", exact="preserved"),
    _spec("pvariance", "statistics", "pvariance(sequence)", "Variance de population.", exact="depends"),
    _spec("pstdev", "statistics", "pstdev(sequence)", "Ecart type de population.", exact="false"),
    _spec("variance", "statistics", "variance(sequence)", "Variance d'echantillon.", exact="depends"),
    _spec("stdev", "statistics", "stdev(sequence)", "Ecart type d'echantillon.", exact="false"),
    _spec("frac", "exact", "frac(numerator, denominator)", "Fraction rationnelle reduite.", exact="true", min_args=2, max_args=2),
)

_REGISTRY: Final = {spec.name: spec for spec in _SPECS}
_CONSTANTS: Final = {"pi": math.pi, "e": math.e, "tau": math.tau}
_BINARY_OPERATORS: Final = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
}
_UNARY_OPERATORS: Final = {ast.UAdd: "+ (unaire)", ast.USub: "- (unaire)"}
_ALLOWED_NODE_TYPES: Final = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)

_UNARY_MATH: Final = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "degrees": math.degrees,
    "radians": math.radians,
}
_STATISTICS: Final = {
    "mean": statistics.mean,
    "fmean": statistics.fmean,
    "median": statistics.median,
    "median_low": statistics.median_low,
    "median_high": statistics.median_high,
    "pvariance": statistics.pvariance,
    "pstdev": statistics.pstdev,
    "variance": statistics.variance,
    "stdev": statistics.stdev,
}


def _error(code: str, message: str) -> MathEngineError:
    return MathEngineError(message, code=code)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, Fraction))


def _number(value: Any, *, name: str = "argument") -> int | float | Fraction:
    if not _is_number(value):
        raise _error("invalid_arguments", f"{name} doit etre un nombre reel.")
    return value


def _integer(value: Any, *, name: str = "argument") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error("invalid_arguments", f"{name} doit etre un entier.")
    return value


def _tree_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_tree_depth(child) for child in children)


def _digits(value: int) -> int:
    return len(str(abs(value))) if value else 1


def _check_number(value: Any, limits: MathLimits) -> int | float | Fraction:
    value = _number(value, name="resultat")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error("numeric_limit", "Le resultat doit etre un nombre fini.")
        return value
    integers = (value.numerator, value.denominator) if isinstance(value, Fraction) else (value,)
    for integer in integers:
        if abs(integer).bit_length() > limits.max_integer_bits:
            raise _error("numeric_limit", "Le resultat depasse la limite de bits.")
        if _digits(integer) > limits.max_output_digits:
            raise _error("numeric_limit", "Le resultat depasse la limite de chiffres.")
    return value


def _sequence(
    args: list[Any],
    *,
    name: str,
    limits: MathLimits,
    minimum: int = 1,
) -> list[int | float | Fraction]:
    if len(args) != 1 or not isinstance(args[0], (list, tuple)):
        raise _error("invalid_arguments", f"{name} attend une liste ou un tuple.")
    values = list(args[0])
    if len(values) < minimum:
        raise _error("domain_error", f"{name} exige au moins {minimum} valeur(s).")
    if len(values) > limits.max_collection_items:
        raise _error("resource_limit", "La serie contient trop de valeurs.")
    return [_number(value, name="valeur de la serie") for value in values]


def _estimated_factorial_bits(n: int) -> int:
    if n < 2:
        return 1
    return max(1, math.ceil(math.lgamma(n + 1) / math.log(2)))


def _estimated_comb_bits(n: int, k: int) -> int:
    if k in (0, n):
        return 1
    estimate = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    ) / math.log(2)
    return max(1, math.ceil(estimate))


def _estimated_perm_bits(n: int, k: int) -> int:
    if k == 0:
        return 1
    estimate = (math.lgamma(n + 1) - math.lgamma(n - k + 1)) / math.log(2)
    return max(1, math.ceil(estimate))


class _Interpreter:
    def __init__(self, limits: MathLimits):
        self.limits = limits
        self.functions_used: list[str] = []
        self.operations: list[str] = []

    def evaluate(self, node: ast.AST) -> int | float | Fraction | list[Any] | tuple[Any, ...]:
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise _error("unsupported_expression", "Seuls les nombres sont autorises.")
            return _check_number(node.value, self.limits)
        if isinstance(node, ast.Name):
            if node.id not in _CONSTANTS:
                raise _error("unsupported_expression", "Nom ou fonction non autorise.")
            return _CONSTANTS[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > self.limits.max_collection_items:
                raise _error("resource_limit", "La collection contient trop de valeurs.")
            values = [self.evaluate(item) for item in node.elts]
            if any(not _is_number(value) for value in values):
                raise _error("unsupported_expression", "Les collections doivent etre numeriques.")
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.UnaryOp):
            symbol = _UNARY_OPERATORS.get(type(node.op))
            if symbol is None:
                raise _error("unsupported_expression", "Operateur unaire non autorise.")
            operand = _number(self.evaluate(node.operand))
            self.operations.append(symbol)
            result = operand if isinstance(node.op, ast.UAdd) else -operand
            return _check_number(result, self.limits)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise _error("unsupported_expression", "Construction syntaxique non autorisee.")

    def _binary(self, node: ast.BinOp) -> int | float | Fraction:
        symbol = _BINARY_OPERATORS.get(type(node.op))
        if symbol is None:
            raise _error("unsupported_expression", "Operateur binaire non autorise.")
        left = _number(self.evaluate(node.left), name="operande gauche")
        right = _number(self.evaluate(node.right), name="operande droite")
        self.operations.append(symbol)

        if isinstance(node.op, ast.Mult):
            if isinstance(left, int) and isinstance(right, int) and left and right:
                if abs(left).bit_length() + abs(right).bit_length() > self.limits.max_integer_bits + 1:
                    raise _error("numeric_limit", "Le produit depasse la limite de bits.")
            result = left * right
        elif isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        else:
            exponent = _integer(right, name="exposant")
            if abs(exponent) > self.limits.max_exponent:
                raise _error("resource_limit", "L'exposant depasse la limite autorisee.")
            if exponent > 0 and isinstance(left, int) and abs(left) > 1:
                if abs(left).bit_length() * exponent > self.limits.max_integer_bits:
                    raise _error("numeric_limit", "La puissance depasse la limite de bits.")
            if exponent and isinstance(left, Fraction):
                magnitude = abs(exponent)
                if (
                    abs(left.numerator).bit_length() * magnitude > self.limits.max_integer_bits
                    or left.denominator.bit_length() * magnitude > self.limits.max_integer_bits
                ):
                    raise _error("numeric_limit", "La puissance depasse la limite de bits.")
            result = left**exponent
        return _check_number(result, self.limits)

    def _call(self, node: ast.Call) -> int | float | Fraction:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise _error("unsupported_expression", "Appel de fonction non autorise.")
        name = node.func.id
        spec = _REGISTRY.get(name)
        if spec is None:
            raise _error("unsupported_expression", "Nom ou fonction non autorise.")
        if len(node.args) < spec.min_args or len(node.args) > spec.max_args:
            raise _error("invalid_arguments", f"Nombre d'arguments invalide pour {name}.")
        if len(node.args) > self.limits.max_function_args:
            raise _error("resource_limit", "Trop d'arguments pour une fonction.")
        args = [self.evaluate(argument) for argument in node.args]
        self.functions_used.append(name)
        return _check_number(self._invoke(name, args), self.limits)

    def _invoke(self, name: str, args: list[Any]) -> int | float | Fraction:
        if name == "abs":
            return abs(_number(args[0]))
        if name == "round":
            value = _number(args[0])
            if len(args) == 1:
                return round(value)
            digits = _integer(args[1], name="digits")
            if abs(digits) > self.limits.max_round_digits:
                raise _error("resource_limit", "Le nombre de decimales est trop grand.")
            return round(value, digits)
        if name in {"floor", "ceil", "trunc"}:
            return getattr(math, name)(_number(args[0]))
        if name in {"sum", "fsum", "prod", "min", "max"}:
            minimum = 0 if name in {"sum", "fsum", "prod"} else 1
            values = _sequence(args, name=name, limits=self.limits, minimum=minimum)
            if name == "fsum":
                return math.fsum(values)
            if name == "min":
                return min(values)
            if name == "max":
                return max(values)
            result: int | float | Fraction = 0 if name == "sum" else 1
            for value in values:
                result = result + value if name == "sum" else result * value
                result = _check_number(result, self.limits)
            return result
        if name in _UNARY_MATH:
            return _UNARY_MATH[name](_number(args[0]))
        if name == "isqrt":
            value = _integer(args[0], name="n")
            return math.isqrt(value)
        if name == "log":
            value = _number(args[0])
            return math.log(value) if len(args) == 1 else math.log(value, _number(args[1]))
        if name == "hypot":
            return math.hypot(*[_number(value) for value in args])
        if name == "atan2":
            return math.atan2(_number(args[0]), _number(args[1]))
        if name in {"gcd", "lcm"}:
            integers = [_integer(value) for value in args]
            return getattr(math, name)(*integers)
        if name == "factorial":
            n = _integer(args[0], name="n")
            if n < 0:
                raise _error("domain_error", "La factorielle exige n >= 0.")
            if n > self.limits.max_factorial:
                raise _error("resource_limit", "La factorielle depasse la limite autorisee.")
            if _estimated_factorial_bits(n) > self.limits.max_integer_bits:
                raise _error("numeric_limit", "La factorielle depasse la limite de bits.")
            return math.factorial(n)
        if name in {"comb", "perm"}:
            n = _integer(args[0], name="n")
            k = _integer(args[1], name="k") if len(args) == 2 else n
            if n < 0 or k < 0 or k > n:
                raise _error("domain_error", f"{name} exige 0 <= k <= n.")
            if n > self.limits.max_combinatorial_n:
                raise _error("resource_limit", "n depasse la limite combinatoire.")
            estimate = _estimated_comb_bits(n, k) if name == "comb" else _estimated_perm_bits(n, k)
            if estimate > self.limits.max_integer_bits:
                raise _error("numeric_limit", "Le resultat combinatoire depasse la limite de bits.")
            return math.comb(n, k) if name == "comb" else math.perm(n, k)
        if name in _STATISTICS:
            minimum = 2 if name in {"variance", "stdev"} else 1
            values = _sequence(args, name=name, limits=self.limits, minimum=minimum)
            return _STATISTICS[name](values)
        if name == "frac":
            numerator = _integer(args[0], name="numerateur")
            denominator = _integer(args[1], name="denominateur")
            if denominator == 0:
                raise _error("division_by_zero", "Le denominateur ne peut pas etre nul.")
            return Fraction(numerator, denominator)
        raise _error("unsupported_expression", "Fonction non autorisee.")


def _validate_tree(tree: ast.AST, limits: MathLimits) -> tuple[int, int]:
    nodes = list(ast.walk(tree))
    if len(nodes) > limits.max_nodes:
        raise _error("resource_limit", "L'expression contient trop de noeuds.")
    try:
        depth = _tree_depth(tree)
    except RecursionError as error:
        raise _error("resource_limit", "L'expression est trop profonde.") from error
    if depth > limits.max_depth:
        raise _error("resource_limit", "L'expression est trop profonde.")
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise _error("unsupported_expression", "Construction syntaxique non autorisee.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _REGISTRY:
                raise _error("unsupported_expression", "Nom ou fonction non autorise.")
            if node.keywords:
                raise _error("unsupported_expression", "Les arguments nommes sont interdits.")
        elif isinstance(node, ast.Name) and node.id not in _CONSTANTS and node.id not in _REGISTRY:
            raise _error("unsupported_expression", "Nom ou fonction non autorise.")
    return len(nodes), depth


def _result_type(value: int | float | Fraction) -> str:
    if isinstance(value, Fraction):
        return "fraction"
    if isinstance(value, int):
        return "integer"
    return "float"


def _json_integer(value: int) -> int | str:
    """Preserve exact integers when a JSON client uses IEEE-754 numbers."""

    return value if abs(value) <= _MAX_JSON_SAFE_INTEGER else str(value)


def _json_result(
    value: int | float | Fraction,
) -> int | float | str | dict[str, int | str]:
    if isinstance(value, Fraction):
        return {
            "numerator": _json_integer(value.numerator),
            "denominator": _json_integer(value.denominator),
        }
    if isinstance(value, int):
        return _json_integer(value)
    return value


def _display(value: int | float | Fraction) -> str:
    if isinstance(value, Fraction):
        return (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        )
    if isinstance(value, int):
        return str(value)
    return format(value, ".15g")


def _approximation(value: int | float | Fraction) -> int | float | None:
    if isinstance(value, int):
        return value if abs(value) <= _MAX_JSON_SAFE_INTEGER else None
    if isinstance(value, float):
        return value
    try:
        approximate = float(value)
    except (OverflowError, ValueError):
        return None
    return approximate if math.isfinite(approximate) else None


class MathEngine:
    """Interpreteur AST pur et reutilisable."""

    def __init__(self, limits: MathLimits | None = None):
        self.limits = limits or MathLimits()
        if not isinstance(self.limits, MathLimits):
            raise TypeError("limits doit etre une instance de MathLimits")

    def evaluate(self, expression: str) -> dict[str, Any]:
        """Calcule une expression autorisee sans ecriture externe."""

        started = time.perf_counter_ns()
        if not isinstance(expression, str):
            raise _error("invalid_expression", "L'expression doit etre une chaine.")
        clean_expression = unicodedata.normalize("NFKC", expression).strip()
        if not clean_expression:
            raise _error("invalid_expression", "L'expression ne peut pas etre vide.")
        if len(clean_expression) > self.limits.max_expression_chars:
            raise _error("resource_limit", "L'expression est trop longue.")
        try:
            tree = ast.parse(clean_expression, mode="eval")
        except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
            code = "resource_limit" if isinstance(error, (MemoryError, RecursionError)) else "invalid_syntax"
            raise _error(code, "Syntaxe mathematique invalide.") from error
        node_count, depth = _validate_tree(tree, self.limits)
        interpreter = _Interpreter(self.limits)
        try:
            raw_value = interpreter.evaluate(tree)
            value = _check_number(raw_value, self.limits)
        except MathEngineError:
            raise
        except ZeroDivisionError as error:
            raise _error("division_by_zero", "Division par zero interdite.") from error
        except OverflowError as error:
            raise _error("numeric_limit", "Le calcul depasse les limites numeriques.") from error
        except (MemoryError, RecursionError) as error:
            raise _error("resource_limit", "Le calcul depasse les ressources autorisees.") from error
        except (ValueError, statistics.StatisticsError) as error:
            raise _error("domain_error", "Arguments hors du domaine mathematique.") from error
        except TypeError as error:
            raise _error("invalid_arguments", "Arguments mathematiques invalides.") from error

        duration_ms = (time.perf_counter_ns() - started) / 1_000_000
        kind = _result_type(value)
        functions_used = list(dict.fromkeys(interpreter.functions_used))
        return {
            "expression": clean_expression,
            "result": _json_result(value),
            "value": _json_result(value),
            "display": _display(value),
            "type": kind,
            "result_type": kind,
            "exact": isinstance(value, (int, Fraction)),
            "approximation": _approximation(value),
            "duration_ms": round(duration_ms, 6),
            "functions": functions_used,
            "functions_used": functions_used,
            "operations": list(interpreter.operations),
            "operations_count": len(interpreter.operations),
            "algorithm": "safe_ast_math",
            "algorithm_version": REGISTRY_VERSION,
            "verification": {
                "status": "policy_validated",
                "method": "safe_ast_registry_execution",
                "scope": "syntaxe, registre autorise et limites de ressources",
                "independent_oracle": False,
                "registry_version": REGISTRY_VERSION,
                "node_count": node_count,
                "depth": depth,
                "resource_limits_checked": True,
                "memory_writes": 0,
            },
        }

    def catalog_entries(self) -> list[dict[str, Any]]:
        return [spec.public() for spec in _SPECS]

    def catalog(self) -> dict[str, Any]:
        entries = self.catalog_entries()
        categories = []
        for category_id, (label, description) in _CATEGORY_INFO.items():
            categories.append(
                {
                    "id": category_id,
                    "label": label,
                    "description": description,
                    "count": sum(entry["category"] == category_id for entry in entries),
                }
            )
        return {
            "version": REGISTRY_VERSION,
            "count": len(entries),
            "learning_levels": [
                {"level": 1, "state": "received"},
                {"level": 2, "state": "observed"},
                {"level": 3, "state": "consolidated"},
                {"level": 4, "state": "operational"},
            ],
            "categories": categories,
            "functions": entries,
            "operators": ["+", "-", "*", "/", "//", "%", "**", "+unaire", "-unaire"],
            "constants": sorted(_CONSTANTS),
            "limits": asdict(self.limits),
        }


_DEFAULT_ENGINE = MathEngine()


def evaluate(expression: str) -> dict[str, Any]:
    """Raccourci module-level utilisant les limites par defaut."""

    return _DEFAULT_ENGINE.evaluate(expression)


def catalog() -> dict[str, Any]:
    return _DEFAULT_ENGINE.catalog()


def catalog_entries() -> list[dict[str, Any]]:
    return _DEFAULT_ENGINE.catalog_entries()


def _assert_json_serializable(value: Any) -> None:
    """Garde interne utile aux tests et aux integrateurs stricts."""

    json.dumps(value, ensure_ascii=False, allow_nan=False)


__all__ = [
    "MathEngine",
    "MathEngineError",
    "MathLimits",
    "REGISTRY_VERSION",
    "catalog",
    "catalog_entries",
    "evaluate",
]
