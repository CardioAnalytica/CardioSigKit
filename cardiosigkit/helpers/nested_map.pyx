# -*- coding: utf-8 -*-
# Cython Source File

# Cython imports
cimport cython
from cpython cimport list, dict, tuple

from iodeeplib.dtype cimport Map # noqa

# Python imports
import json
from bson import ObjectId
from iodeeplib.helper import NumpyArrayEncoder # noqa

cdef class NestedMap:

    def __cinit__(self, data=None):
        if data is None:
            data = Map()
        elif type(data) is dict:
            data = Map(data)
        assert type(data) is Map, f"Expected type of data to be Map got {type(data)}"
        self.data = data

        for key in data.get_keys():
            value = data[key]
            if type(value) in (dict, Map):
                self.__setattr__(key, NestedMap(value))
            else:
                try:
                    self.__setattr__(key, value)
                except AttributeError:
                    print(key)
                    print(value)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __setitem__(self, key, value):
        self.data[key] = value
        if type(value) in (dict, Map):
            self.__setattr__(key, NestedMap(value))
        else:
            try:
                self.__setattr__(key, value)
            except AttributeError:
                print(key)
                print(value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, str path):
        cdef:
            object      leaf
            list        keys
            Py_ssize_t  i
        keys = path.split(".")
        leaf = self.data
        for i in range(len(keys)):
            if keys[i] not in leaf:
                raise KeyError(f"Key {keys[i]} not found")
            leaf = leaf[keys[i]]
        return leaf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __repr__(self):
        return self.dump(format="json")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object dump(self, str format="dict"):
        assert format in ["dict", "json"], "Unrecognized output format"
        cdef:
            dict    dict_output
            object  key
            object  data
        dict_output = {}
        for key, value in self.items():
            if type(value) == Map:
                dict_output[key] = value.dump()
            elif type(value) == ObjectId:
                dict_output[key] = str(value)
            else:
                dict_output[key] = value
        if format == "dict":
            return dict_output
        elif format == "json":
            return json.dumps(dict_output, indent=2, cls=NumpyArrayEncoder, default=str)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint has(self, str path):
        cdef:
            object      leaf
            list        keys
            Py_ssize_t  i
        keys = path.split(".")
        leaf = self.data
        for i in range(len(keys)):
            if keys[i] not in leaf:
                return 0
            leaf = leaf[keys[i]]
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list keys(self):
        return self.data.get_keys()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object items(self):
        return iter(self.data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __len__(self):
        return len(self.data)