# -*- coding: utf-8 -*-
# Cython Source File

# Cython imports
cimport numpy as np
cimport cython
from cpython cimport list, dict
from libcpp.vector cimport vector
from libc.math cimport ceil

# Python imports
import numpy as np
import copy
import os.path, pkgutil
import pathlib

cdef class Parser:
    @staticmethod
    cdef readonly object c_is_file_valid(str filepath):
        pass

    @staticmethod
    def is_file_valid(filepath:str) -> object:
        pass

    cpdef dict parse(self, str filepath, object fd=None, object data=None):
        pass


@cython.final
cdef class FileParser:
    """
    Class level comments
    """

    def __cinit__(self):
        """
        Cython constructor comments
        @return:
        """
        self.parsers = {}
        self.load_parsers()
        self.active_parsers = {}

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void load_parsers(self) except *:
        """

        @return: 
        """
        cdef:
            str pkgpath
            str base_import_path
            list excludes
            list packages
            list registry
            dict parser
            object module

        pkgpath = os.path.dirname(__file__)
        excludes = ['file_parser', 'parser']
        base_import_path = 'ecgdatakit.parsers'
        self.parsers = {}
        packages = [name for _, name, _ in pkgutil.iter_modules([pkgpath]) if name not in excludes]
        for package in packages:
            module = __import__(f"{base_import_path}.{package}", fromlist=[package])
            registry = module.__registry__
            for entry in registry:
                if entry['extension'] not in self.parsers:
                    self.parsers[entry['extension']] = []
                parser = {}
                parser['format'] = entry['format']
                parser['class'] = entry['parser']
                parser['package'] = package
                parser['import_path'] = f"{base_import_path}.{package}"
                self.parsers[entry['extension']].append(parser)

    cpdef dict get_parsers(self):
        """

        @return: 
        """
        return self.parsers

    cpdef list get_supported_formats(self):
        """

        @return: 
        """
        cdef:
            list formats = []
            list entry
            Py_ssize_t i = 0
            str key
            list value

        for key, value in self.parsers.items():
            for i in range(len(value)):
                entry = [key, value[i]['format']]
            formats.append(entry)
        return formats


    cdef object load_class(self, dict reg):
        """
        
        @param reg: 
        @return: 
        """
        class_name = reg['class']
        module_name = reg['import_path']
        module = __import__(module_name, fromlist=[class_name])
        clazz = getattr(module, class_name)
        if not issubclass(clazz, Parser):
            raise ValueError("Attempted to load not Parser class")
        return clazz

    cpdef void set_format(self, str format) except *:
        """

        @param format: 
        @return: 
        """
        cdef:
            str         class_name
            str         module_name
            object      module
            Py_ssize_t  i
            str         key
            list        value
        i = 0
        for key, value in self.parsers.items():
            for i in range(len(value)):
                if value[i]['format'] == format:
                    clazz = self.load_class(value[i])
                    self.active_parser = clazz()
                    print(self.active_parser)
                    return
        raise ValueError("Could not find specified format")


    cpdef void unfreeze(self) except *:
        """
        
        @return: 
        """
        self.active_parser = None


    cpdef dict parse(self, str filepath):
        """
        
        @param filepath: 
        @return: 
        """
        cdef:
            Py_ssize_t  i
            str         key
            list        value
            object      clazz
            str         file_extension
            list        entries
            object      parser_res
        file_extension = pathlib.Path(filepath).suffix.replace('.', '')
        if self.active_parser is not None:
            return self.active_parser.parse(filepath)
        if len(self.active_parsers) > 0 and file_extension in self.active_parsers:
            entries = self.active_parsers[file_extension]
            for i in range(len(entries)):
                clazz = entries[i]
                parser_res = clazz.is_file_valid(filepath)
                if parser_res != False:
                    assert isinstance(parser_res, tuple) and len(parser_res) == 2, "Expected a 2 items tuple return value for parser validation function"
                    return clazz.parse(filepath, parser_res[0], parser_res[1])
        clazz = self.parser_factory(filepath, file_extension)
        assert clazz is not None, "Failed to find suitable parser"
        return clazz[0].parse(filepath, clazz[1], clazz[2])


    cdef object parser_factory(self, str filepath, str file_extension):
        """
        
        @param filepath: 
        @param file_extension: 
        @return: 
        """
        cdef:
            Py_ssize_t  i
            str         key
            list        value
            list        entries
            object      clazz
        if file_extension not in self.parsers:
            return None
        entries = self.parsers[file_extension]
        for i in range(len(entries)):
            entry = entries[i]
            clazz = self.load_class(entry)
            parser_res = clazz.is_file_valid(filepath)
            if parser_res != False:
                clazz = clazz()
                if file_extension not in self.active_parsers:
                    self.active_parsers[file_extension] = []
                self.active_parsers[file_extension].append(clazz)
                return clazz, parser_res[0], parser_res[1]
        return None