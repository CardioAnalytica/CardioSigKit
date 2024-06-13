# -*- coding: utf-8 -*-
# Cython Declaration File

cimport cython
from cpython cimport list, dict

from collections import OrderedDict

cpdef object find_xml_dict_tag(object xml2dict_doc, str tag)
cpdef object read_xml_dict_path(object xml2dict_doc, str path)