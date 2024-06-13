# -*- coding: utf-8 -*-
# Cython Source File

cimport cython
from cpython cimport list, dict

from collections import OrderedDict

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object find_xml_dict_tag(object xml2dict_doc, str tag):
    cdef:
        str         k
        object      v
        object      rec
        list        results
    results = []
    if isinstance(xml2dict_doc, dict):
        for k, v in xml2dict_doc.items():
            if k.lower() == tag.lower():
                results.append(v)
            else:
                rec = find_xml_dict_tag(v, tag)
                if rec is not None:
                    if isinstance(rec, list):
                        results.extend(rec)
                    else:
                        results.append(rec)
    elif isinstance(xml2dict_doc, list):
        for i in range(len(xml2dict_doc)):
            rec = find_xml_dict_tag(xml2dict_doc[i], tag)
            if rec is not None:
                if isinstance(rec, list):
                    results.extend(rec)
                else:
                    results.append(rec)
    else:
        return None
    return results[0] if len(results) == 1 else None if len(results) == 0 else results


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object read_xml_dict_path(object xml2dict_doc, str path):
    cdef:
        list    parts
    if xml2dict_doc is None:
        return None
    parts = path.split("/")
    if len(parts) >= 1 and parts[0] not in xml2dict_doc:
        return None
    if len(parts) == 1:
        return xml2dict_doc[parts[0]]
    else:
        return read_xml_dict_path(xml2dict_doc[parts[0]], "/".join(parts[1:]))