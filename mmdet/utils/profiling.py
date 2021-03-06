B
    ??a?  ?               @   s   d Z ddlZG dd? d?ZdS )a=  This module defines the :class:`NiceRepr` mixin class, which defines a
``__repr__`` and ``__str__`` method that only depend on a custom ``__nice__``
method, which you must define. This means you only have to overload one
function instead of two.  Furthermore, if the object defines a ``__len__``
method, then the ``__nice__`` method defaults to something sensible, otherwise
it is treated as abstract and raises ``NotImplementedError``.

To use simply have your object inherit from :class:`NiceRepr`
(multi-inheritance should be ok).

This code was copied from the ubelt library: https://github.com/Erotemic/ubelt

Example:
    >>> # Objects that define __nice__ have a default __str__ and __repr__
    >>> class Student(NiceRepr):
    ...    def __init__(self, name):
    ...        self.name = name
    ...    def __nice__(self):
    ...        return self.name
    >>> s1 = Student('Alice')
    >>> s2 = Student('Bob')
    >>> print(f's1 = {s1}')
    >>> print(f's2 = {s2}')
    s1 = <Student(Alice)>
    s2 = <Student(Bob)>

Example:
    >>> # Objects that define __len__ have a default __nice__
    >>> class Group(NiceRepr):
    ...    def __init__(self, data):
    ...        self.data = data
    ...    def __len__(self):
    ...        return len(s