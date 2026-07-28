"""TEMPORARY -- deliberate violation, proving issue #2's compileall floor job bites.

PEP 695 type-parameter defaults are 3.13+ syntax, so this is a SyntaxError on the declared floor.
Nothing imports this module -- that is the point: pytest would never load it, and clients/viewer is
never installed headless, so only the compile job can catch it.
"""


class Box[T = int]:
    pass
