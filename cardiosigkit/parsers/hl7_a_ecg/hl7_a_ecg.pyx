# -*- coding: utf-8 -*-
# Cython Source File

# Cython imports
cimport numpy as np
cimport cython
from cpython cimport list, dict
from libcpp.vector cimport vector
from libc.math cimport ceil
from cpython.datetime cimport datetime
from cardiosigkit.helpers cimport xmlutils
from cardiosigkit.parsers cimport Parser

# Python imports
import numpy as np
import copy
import uuid as uuidlib
import os
import xmltodict

cdef datetime parse_str_datetime(str value):
    """
    
    @param value: 
    @return: 
    """
    if value is None:
        return None
    if "." in value:
        value = value.split(".")[0]
    return datetime.strptime(value, '%Y%m%d%H%M%S').replace(microsecond=1)


cdef class HL7aECG(Parser):
    """
    Class level comments
    """

    def __cinit__(self):
        """

        @return:
        """
        self.leads_name = ["MDC_ECG_LEAD_I", "MDC_ECG_LEAD_II", "MDC_ECG_LEAD_III", "MDC_ECG_LEAD_AVR", "MDC_ECG_LEAD_AVL",
                      "MDC_ECG_LEAD_AVF", "MDC_ECG_LEAD_V1", "MDC_ECG_LEAD_V2", "MDC_ECG_LEAD_V3", "MDC_ECG_LEAD_V4",
                      "MDC_ECG_LEAD_V5", "MDC_ECG_LEAD_V6", "MDC_ECG_LEAD_IIc"]


    @staticmethod
    cdef readonly object c_is_file_valid(str filepath):
        """
        
        @param filepath: 
        @return: 
        """
        if os.path.isfile(filepath) and filepath.lower().endswith(".xml"):
            try:
                fd = open(filepath, "rb")
                doc = xmltodict.parse(fd.read())
                if len(doc) == 1 and list(doc.keys())[0] == "AnnotatedECG":
                    return fd, doc
            except Exception:
                pass
        return False


    @staticmethod
    def is_file_valid(filepath:str) -> bool:
        return HL7aECG.c_is_file_valid(filepath)


    cdef void read_patient_data(self, object doc, dict record) except *:
        """
        
        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            object  node
            object  node_tmp
            str     gender
            str     birthdate
            dict    entry
        entry = {}
        node = xmlutils.find_xml_dict_tag(doc, "subject")
        if node is not None:
            node_tmp = xmlutils.find_xml_dict_tag(node, "administrativeGenderCode")
            if node_tmp is not None:
                gender = xmlutils.read_xml_dict_path(node_tmp, "@code")
                gender = "Male" if gender == "M" else "Female" if gender == "F" else "UN"
                if gender is not None:
                   entry['sex'] = gender
            node_tmp = xmlutils.find_xml_dict_tag(node, "birthTime")
            if node_tmp is not None:
                birthdate = xmlutils.read_xml_dict_path(node_tmp, "@value")
                if birthdate is not None:
                    birthdate = birthdate[:4]+"/"+birthdate[4:6]+"/"+birthdate[6:]
                    entry['birth_date'] = birthdate
        record['patient'] = entry


    cdef void read_ecg_data(self, object doc, dict record) except *:
        """
        
        @param doc: 
        @param record: 
        @return: 
        """
        record["ecg"]["uuid"] = str(uuidlib.uuid4())
        record["ecg"]["dataset_ecg_uuid"] = xmlutils.read_xml_dict_path(doc, "AnnotatedECG/id/@root")
        record["ecg"]["anonymized"] = True


    cdef void read_recording_data(self, object doc, dict record) except *:
        """
        
        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            str     visit
            dict    event
        event = {}
        visit = xmlutils.read_xml_dict_path(xmlutils.find_xml_dict_tag(doc, 'TimepointEvent'), "code/@code")
        if visit is not None and len(visit) > 0:
            event['visit'] = {}
            event['visit']['name'] = visit
        event['started_at'] = parse_str_datetime(
            xmlutils.read_xml_dict_path(doc, "AnnotatedECG/effectiveTime/low/@value"))
        event['ended_at'] = parse_str_datetime(
            xmlutils.read_xml_dict_path(doc, "AnnotatedECG/effectiveTime/high/@value"))
        record['recording'] = event


    cdef void parse_signals_series(self, object doc, dict buffer) except *:
        """
        
        @param node: 
        @param leads: 
        @return: 
        """
        cdef:
            list    nodes
            list    signal
            str     code
            list    raw_digits
            str     raw_digit
            object  node

        nodes = xmlutils.read_xml_dict_path(doc, "sequenceSet/component")
        for i, node in enumerate(nodes):
            try:
                code = xmlutils.read_xml_dict_path(node, "sequence/code/@code")
                if code in self.leads_name:
                    signal = []
                    raw_digits = xmlutils.read_xml_dict_path(node, "sequence/value/digits").split(" ")
                    code = code.replace("MDC_ECG_LEAD_", "").replace("AV", "aV")
                    for raw_digit in raw_digits:
                        elem = raw_digit.replace("\n", "")
                        if len(elem) > 0:
                            signal.append(int(float(raw_digit)))
                    buffer[code] = signal
            except:
                raise RuntimeError(f"Failed to parse signal node with index {i}")


    cdef void read_signals_data(self, object doc, dict record) except *:
        """
        
        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            object      node
            object      nodes
            str         code
            dict        leads
            Py_ssize_t  i
            dict        beats

        # Read leads
        leads = {}
        nodes = xmlutils.read_xml_dict_path(xmlutils.find_xml_dict_tag(doc, "series"), "component")
        if isinstance(nodes, list):
            for i in range(len(nodes)):
                self.parse_signals_series(nodes[i], leads)
        else:
            self.parse_signals_series(nodes, leads)
        record["ecg"]["leads"] = leads
        record["ecg"]["timepoints"] = len(leads[list(leads.keys())[0]])

        # Read representative beats
        beats = {}
        node = xmlutils.find_xml_dict_tag(doc, "derivedSeries")
        code = xmlutils.read_xml_dict_path(node, "code/@code")
        if code == "REPRESENTATIVE_BEAT":
            nodes = xmlutils.read_xml_dict_path(node, "component")
            self.parse_signals_series(nodes, beats)
            record["ecg"]["representative_beats"] = beats


    cdef void read_annotations_data(self) except *:
        pass


    cpdef dict parse(self, str filepath, object fd=None, object doc=None):
        """
        
        @param filepath: 
        @param fd: 
        @param doc: 
        @return: 
        """
        cdef:
            dict    record
        record = {}
        record['ecg'] = {}
        assert os.path.exists(filepath), "File does not exists"
        if fd is None:
            fd = open(filepath, "rb")
        if doc is None:
            doc = xmltodict.parse(fd.read())
        self.read_ecg_data(doc, record)
        self.read_recording_data(doc, record)
        self.read_patient_data(doc, record)
        self.read_signals_data(doc, record)
        record['ecg']['filepath'] = filepath
        fd.close()
        return record