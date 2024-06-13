# -*- coding: utf-8 -*-
# Cython Declaration File

# Cython imports
cimport numpy as np
cimport cython
from cpython cimport list, dict
from cardiosigkit.parsers cimport Parser

# Python imports
import numpy as np

ctypedef fused iterable:
    list
    tuple

ctypedef fused iterable_array:
    iterable
    np.ndarray

cdef class HL7aECG(Parser):
    cdef:
        list    leads_name

    @staticmethod
    cdef readonly object c_is_file_valid(str filepath)
    cdef void read_patient_data(self, object doc, dict record) except *
    cdef void read_ecg_data(self, object doc, dict record) except *
    cdef void read_recording_data(self, object doc, dict record) except *
    cdef void parse_signals_series(self, object doc, dict buffer) except *
    cdef void read_signals_data(self, object doc, dict record) except *
    cdef void read_annotations_data(self) except *
    cpdef dict parse(self, str filepath, object fd=*, object doc=*)