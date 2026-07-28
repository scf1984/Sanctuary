"""TEMPORARY -- deliberate violation, proving issue #2's version gates actually bite.

`@classmethod` stacked on `@property` is the exact construct the 2017-2023 prototype used. It
chains on 3.12 (deprecated) and was removed in 3.13, where the attribute access returns the
property object instead of the value -- silently, which is how the prototype stopped running with
nothing to record it.

Expected: test (py3.12) passes, test (py3.14) fails. This branch is never merged.
"""


class Prototype:
    @classmethod
    @property
    def value(cls):
        return 1


def test_classmethod_property_chaining_still_works():
    assert Prototype.value == 1
