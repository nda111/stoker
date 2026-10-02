"""Name to object lookup, for selecting an implementation by string.

A :class:`Registry` lets a config field hold a name while the object it refers
to is declared elsewhere, which keeps the thing being chosen next to its own
definition rather than in a central dispatch table.

    MODELS = Registry[type[nn.Module]]()

    @MODELS.register('resnet18')
    class ResNet18(nn.Module):
        ...

    model = MODELS[config.model]()
"""
from collections.abc import Callable
from typing import Any, Generic, overload
from typing_extensions import TypeVar


_T_REGISTERED_OBJECT = TypeVar('_T_REGISTERED_OBJECT', default=Any)


class Registry(Generic[_T_REGISTERED_OBJECT]):
    """A mapping from names to objects, written to by :meth:`register`.

    Parameterise it with what it holds, as in ``Registry[type[nn.Module]]()``,
    so that lookups are typed. Each instance is independent; make one per kind
    of thing being registered.
    """

    def __init__(self):
        self._items: dict[str, _T_REGISTERED_OBJECT] = {}

    @overload
    def register(
        self,
        name: str,
        *,
        overwrite: bool = False,
    ) -> Callable[[_T_REGISTERED_OBJECT], _T_REGISTERED_OBJECT]:
        ...

    @overload
    def register(
        self,
        obj: _T_REGISTERED_OBJECT,
        name: str,
        *,
        overwrite: bool = False,
    ) -> _T_REGISTERED_OBJECT:
        ...

    def register(
        self,
        obj_or_name: _T_REGISTERED_OBJECT | str,
        name: str | None = None,
        *,
        overwrite: bool = False,
    ) -> (
        _T_REGISTERED_OBJECT
        | Callable[[_T_REGISTERED_OBJECT], _T_REGISTERED_OBJECT]
    ):
        """Add an object under a name, as a decorator or a direct call.

        Pass a name alone to get a decorator; pass an object and a name to
        register something you did not define.

            @MODELS.register('resnet18')
            class ResNet18(nn.Module):
                ...

            MODELS.register(timm_resnet18, 'resnet18_timm')

        Either form returns the object unchanged, so decorating does not wrap
        or alter what it is applied to.

        Args:
            obj_or_name: The name, for the decorator form, or the object to
                register.
            name: The name, when an object was given. Required in that form.
            overwrite: Allow replacing an existing entry. Off by default so
                that a duplicate name is reported rather than silently
                shadowing, which matters when registration happens as a side
                effect of importing.

        Returns:
            The object, or the decorator that registers and returns it.

        Raises:
            TypeError: If an object was given with no name.
            KeyError: If the name is taken and ``overwrite`` is false.
        """
        if isinstance(obj_or_name, str):
            def decorator(
                obj: _T_REGISTERED_OBJECT,
            ) -> _T_REGISTERED_OBJECT:
                return self._register(
                    obj,
                    obj_or_name,
                    overwrite=overwrite,
                )

            return decorator

        if name is None:
            raise TypeError('name is required')

        return self._register(
            obj_or_name,
            name,
            overwrite=overwrite,
        )

    def _register(
        self,
        obj: _T_REGISTERED_OBJECT,
        name: str,
        *,
        overwrite: bool,
    ) -> _T_REGISTERED_OBJECT:
        if not overwrite and name in self._items:
            raise KeyError(f'{name!r} is already registered')

        self._items[name] = obj
        return obj

    def get(self, name: str) -> _T_REGISTERED_OBJECT:
        """Look up a name.

        Note that unlike :meth:`dict.get` this takes no fallback and raises on
        a missing name. It is identical to ``registry[name]``.

        Raises:
            KeyError: If nothing is registered under that name.
        """
        return self._items[name]

    def __getitem__(self, name: str) -> _T_REGISTERED_OBJECT:
        """Look up a name. Same as :meth:`get`.

        Raises:
            KeyError: If nothing is registered under that name.
        """
        return self._items[name]
