# -*- coding: utf-8 -*-
# Cython Declaration File

# Cython imports
cimport numpy as np
cimport cython
from cpython cimport list, dict

# Python imports
import numpy as np

cdef class Parser:
    @staticmethod
    cdef readonly object c_is_file_valid(str filepath)
    cpdef dict parse(self, str filepath, object fd=*, object data=*)


cdef class FileParser:
    cdef:
        dict   parsers
        dict   active_parsers
        Parser active_parser

    cdef void load_parsers(self) except *
    cpdef void set_format(self, str format) except *
    cpdef void unfreeze(self) except *
    cpdef list get_supported_formats(self)
    cpdef dict get_parsers(self)
    cdef object load_class(self, dict reg)
    cpdef dict parse(self, str filepath)
    cdef object parser_factory(self, str filepath, str file_extension)