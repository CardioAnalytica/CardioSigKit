import struct

cimport cython
from cpython cimport list
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bytes serialize_lead_values(list values):
    cdef bytes binary_data = b""
    cdef object num
    for num in values:
        if isinstance(num, int):
            binary_data += b"I" + struct.pack('>i', num)
        elif isinstance(num, float):
            binary_data += b"D" + struct.pack('>d', num)
    return binary_data

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list deserialize_lead_values(bytes binary_data):
    cdef list deserialized_numbers = []
    cdef int pointer = 0
    cdef str type_identifier
    cdef object num
    while pointer < len(binary_data):
        type_identifier = binary_data[pointer:pointer + 1].decode('utf-8')
        pointer += 1
        if type_identifier == "I":
            num = struct.unpack('>i', binary_data[pointer:pointer + 4])[0]
            pointer += 4
        elif type_identifier == "D":
            num = struct.unpack('>d', binary_data[pointer:pointer + 8])[0]
            pointer += 8
        deserialized_numbers.append(num)
    return deserialized_numbers