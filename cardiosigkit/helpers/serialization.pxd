cimport cython
from cpython cimport list

cpdef bytes serialize_lead_values(list values)
cpdef list deserialize_lead_values(bytes binary_data)