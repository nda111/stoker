"""Registering objects under a name and reading them back."""
import pytest

from stoker.registry import Registry


def test_register_as_a_decorator_returns_the_object_unchanged():
    registry = Registry()

    @registry.register('thing')
    class Thing:
        pass

    assert registry['thing'] is Thing
    assert Thing.__name__ == 'Thing'


def test_register_as_a_call_names_an_existing_object():
    registry = Registry()
    value = object()

    returned = registry.register(value, 'value')

    assert registry.get('value') is value
    assert returned is value


def test_get_and_getitem_agree():
    registry = Registry()
    registry.register(1, 'one')

    assert registry.get('one') == registry['one'] == 1


def test_an_unknown_name_raises():
    registry = Registry()

    for lookup in [lambda: registry['missing'], lambda: registry.get('missing')]:
        with pytest.raises(KeyError):
            lookup()


def test_a_duplicate_name_is_rejected():
    registry = Registry()
    registry.register(1, 'one')

    with pytest.raises(KeyError, match='already registered'):
        registry.register(2, 'one')

    assert registry['one'] == 1


def test_overwrite_allows_replacing():
    registry = Registry()
    registry.register(1, 'one')
    registry.register(2, 'one', overwrite=True)

    assert registry['one'] == 2


def test_a_call_without_a_name_is_an_error():
    registry = Registry()

    with pytest.raises(TypeError, match='name is required'):
        registry.register(object())


def test_registries_are_independent():
    models = Registry()
    optimizers = Registry()

    models.register(1, 'shared-name')
    optimizers.register(2, 'shared-name')

    assert models['shared-name'] == 1
    assert optimizers['shared-name'] == 2


def test_a_string_can_be_registered_by_the_call_form():
    # The decorator form is chosen when the first argument is a str, so
    # registering a string value needs the two argument form to be unambiguous.
    registry = Registry()
    registry.register('a value', 'key')

    assert registry['key'] == 'a value'
