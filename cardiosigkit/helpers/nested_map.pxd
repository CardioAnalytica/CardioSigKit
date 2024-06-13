# -*- coding: utf-8 -*-
# Cython Declaration File

# Cython imports

from iodeeplib.dtype cimport Map # noqa

# Python imports

cdef class NestedMap:
    cdef:
        Map     data
        dict    __dict__

    cpdef bint      has(self, str path)
    cpdef list      keys(self)
    cpdef object    items(self)
    cpdef object    dump(self, str format=*)