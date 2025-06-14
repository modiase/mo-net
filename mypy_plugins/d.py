from typing import Callable, Optional, Type as TypingType

from mypy.checker import TypeChecker
from mypy.nodes import MemberExpr
from mypy.plugin import AnalyzeTypeContext, AttributeContext, MethodContext, Plugin
from mypy.types import AnyType, Instance, Type, TypeOfAny, get_proper_type


class DTypePlugin(Plugin):
    """Plugin to make D[T] behave like T for attribute access and operations."""

    def get_type_analyze_hook(
        self, fullname: str
    ) -> Optional[Callable[[AnalyzeTypeContext], Type]]:
        if fullname == "mo_net.protos.D":
            return self._handle_d_type
        return None

    def _handle_d_type(self, ctx: AnalyzeTypeContext) -> Type:
        return ctx.type

    def get_attribute_hook(
        self, fullname: str
    ) -> Optional[Callable[[AttributeContext], Type]]:
        if fullname.startswith("mo_net.protos.D."):
            return self._handle_d_attribute
        return None

    def get_method_hook(self, fullname: str) -> Optional[Callable[[MethodContext], Type]]:
        if fullname in ("mo_net.protos.D.__add__", "mo_net.protos.D.__radd__"):
            return self._handle_d_op
        return None

    def _get_wrapped_type(self, type_in: Instance) -> Optional[Type]:
        if type_in.type.fullname != "mo_net.protos.D" or not type_in.args:
            return None
        return get_proper_type(type_in.args[0])

    def _handle_d_op(self, ctx: MethodContext) -> Type:
        """Handle binary operations on D[T] types."""
        return ctx.type

    def _handle_d_attribute(self, ctx: AttributeContext) -> Type:
        """Handle attribute access on D[T] types."""
        if not isinstance(ctx.type, Instance):
            return ctx.default_attr_type

        wrapped_type = self._get_wrapped_type(ctx.type)
        if not wrapped_type:
            return ctx.default_attr_type

        if not isinstance(ctx.context, MemberExpr):
            return AnyType(TypeOfAny.from_error)

        return ctx.api.get_attribute(wrapped_type, ctx.context.name, ctx.context)  # type: ignore[attr-defined]


def plugin(_version: str) -> TypingType[Plugin]:
    return DTypePlugin