"""Structured, non-fatal error reporting.

Parsers never raise into the GUI. They return a :class:`ParseResult` carrying
either a value, a list of :class:`Issue` objects, or both — a file can be
partially readable and still useful.

Every issue has a short human-readable ``message`` for the UI and an optional
``detail`` (traceback, offending line) that only ever reaches the log file.
"""

from __future__ import annotations

import functools
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    """A single problem encountered while reading or validating a file."""

    code: str
    message: str
    detail: str = ""
    severity: Severity = Severity.WARNING
    source: str | None = None

    def __str__(self) -> str:
        where = f"[{self.source}] " if self.source else ""
        return f"{where}{self.message}"


@dataclass
class ParseResult(Generic[T]):
    """Outcome of a parse attempt: a value, some issues, or both."""

    value: T | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def failed(self) -> bool:
        return self.value is None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @classmethod
    def success(cls, value: T, issues: list[Issue] | None = None) -> "ParseResult[T]":
        return cls(value=value, issues=list(issues or []))

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        detail: str = "",
        source: str | None = None,
    ) -> "ParseResult[T]":
        return cls(
            value=None,
            issues=[
                Issue(
                    code=code,
                    message=message,
                    detail=detail,
                    severity=Severity.ERROR,
                    source=source,
                )
            ],
        )


def safe_parser(source_name: str) -> Callable:
    """Decorator guaranteeing a parser returns a ParseResult instead of raising.

    Any unexpected exception becomes an ERROR issue whose ``detail`` holds the
    full traceback for the log, while the UI only sees a one-line message.
    """

    def decorator(func: Callable[..., ParseResult[T]]) -> Callable[..., ParseResult[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> ParseResult[T]:
            try:
                return func(*args, **kwargs)
            except FileNotFoundError:
                return ParseResult.failure(
                    code=f"{source_name.lower()}_missing",
                    message=f"{source_name} was not found.",
                    source=source_name,
                )
            except UnicodeDecodeError as exc:
                return ParseResult.failure(
                    code=f"{source_name.lower()}_encoding",
                    message=f"{source_name} is not a readable text file.",
                    detail=str(exc),
                    source=source_name,
                )
            except Exception as exc:  # noqa: BLE001 - deliberate GUI firewall
                return ParseResult.failure(
                    code=f"{source_name.lower()}_parse_failed",
                    message=f"{source_name} was found but could not be parsed.",
                    detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    source=source_name,
                )

        return wrapper

    return decorator
