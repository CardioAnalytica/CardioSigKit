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
from datetime import datetime, timedelta
import xmltodict
from iodeeplib.std import printf
from tqdm import tqdm
from collections import OrderedDict
import csv
from cardiosigkit.thirdparty.sierraecg.lib import assert_version, get_node, read_file
from defusedxml import minidom

class XMLField:
    def __init__(self, root_node_name, field_path, dtype):
        supported_dtypes = ("str", "int", "float", "bool")
        assert dtype in supported_dtypes, f"Invalid dtype, supported values are {supported_dtypes}"
        self.__root_node_name = root_node_name
        self.__field_path = field_path
        self.__dtype = dtype

    def get_value(self, doc):
        extracted_doc = xmlutils.find_xml_dict_tag(doc, self.__root_node_name)
        if extracted_doc is None:
            return None
        data = xmlutils.read_xml_dict_path(extracted_doc, self.__field_path)
        if data is not None and data != "":
            try:
                if self.__dtype == "str":
                    return str(data)
                elif self.__dtype == "int":
                    return int(data)
                elif self.__dtype == "float":
                    return float(data)
                elif self.__dtype == "bool":
                    return data.lower() in ("yes", "true", "t", "1")
                else:
                    return data
            except ValueError as e:
                print("Error with field", self.__field_path)
                raise e
        return None

cdef dict FIELDS = {
    "patient": {
        "patient_id": XMLField("generalpatientdata", "patientid", "str"),
        "age": XMLField("generalpatientdata", "age/years", "int"),
        "sex": XMLField("generalpatientdata", "sex", "str"),
        "weight": XMLField("generalpatientdata", "kg", "float"),
        "height": XMLField("generalpatientdata", "cm", "float"),
    },
    "recording": {
        "start_date": XMLField("dataacquisition", "@date", "str"),
        "start_time": XMLField("dataacquisition", "@time", "str"),
        "duration": XMLField("parsedwaveforms", "@durationperchannel", "int")
    },
    "signal_characteristics": [
        {
            "origin": XMLField("dataacquisition", "machine/#text", "str"),
            "sampling_rate": XMLField("signalcharacteristics", "samplingrate", "int"),
            "resolution": XMLField("signalcharacteristics", "resolution", "int"),
            "hipass": XMLField("signalcharacteristics", "hipass", "float"),
            "lowpass": XMLField("signalcharacteristics", "lowpass", "float"),
            "acsetting": XMLField("signalcharacteristics", "acsetting", "int"),
            "notch_filtered": XMLField("signalcharacteristics", "notchfiltered", "bool"),
            "notch_filter_freqs": XMLField("signalcharacteristics", "notchfilterfreqs", "int"),
            "acquisition_type": XMLField("signalcharacteristics", "acquisitiontype", "str"),
            "bits_per_sample": XMLField("signalcharacteristics", "bitspersample", "int"),
            "signal_offset": XMLField("signalcharacteristics", "signaloffset", "int"),
            "signal_signed": XMLField("signalcharacteristics", "signalsigned", "bool"),
            "number_channels_allocated": XMLField("signalcharacteristics", "numberchannelsallocated", "int"),
            "number_channels_valid": XMLField("signalcharacteristics", "numberchannelsvalid", "int"),
            "electrode_placement": XMLField("signalcharacteristics", "electrodeplacement", "str"),
            "order_position_in_all_characteristics": 0
        },
        {
            "data_encoding": XMLField("parsedwaveforms", "@dataencoding", "str"),
            "compression": XMLField("parsedwaveforms", "@compression", "str"),
            "number_of_leads": XMLField("parsedwaveforms", "@numberofleads", "int"),
            "duration": XMLField("parsedwaveforms", "@durationperchannel", "int"),
            "sampling_rate": XMLField("parsedwaveforms", "@samplespersecond", "int"),
            "resolution": XMLField("parsedwaveforms", "@resolution", "int"),
            "signal_offset": XMLField("parsedwaveforms", "@signaloffset", "int"),
            "signal_signed": XMLField("parsedwaveforms", "@signalsigned", "bool"),
            "bits_per_sample": XMLField("parsedwaveforms", "@bitspersample", "int"),
            "hipass": XMLField("parsedwaveforms", "@hipass", "float"),
            "lowpass": XMLField("parsedwaveforms", "@lowpass", "float"),
            "notch_filtered": XMLField("parsedwaveforms", "@notchfiltered", "bool"),
            "notch_filter_freqs": XMLField("parsedwaveforms", "@notchfilterfreqs", "int"),
            "artifact_filtered": XMLField("parsedwaveforms", "@artfiltered", "bool"),
            "waveform_modified": XMLField("parsedwaveforms", "@waveformmodified", "bool"),
            "origin": XMLField("parsedwaveforms", "@modifiedby", "str"),
            "upsampled": XMLField("parsedwaveforms", "@upsampled", "bool"),
            "upsampling_method": XMLField("parsedwaveforms", "@upsamplemethod", "str"),
            "downsampled": XMLField("parsedwaveforms", "@downsampled", "bool"),
            "order_position_in_all_characteristics": 1
        }
    ],
    "annotations": {
        "expert": {
            "interpretation_date": XMLField("interpretation", "@date", "str"),
            "interpretation_time": XMLField("interpretation", "@time", "str"),
            "criteria_version": XMLField("interpretation", "@criteriaversion", "str"),
            "criteria_version_date": XMLField("interpretation", "@criteriaversiondate", "str"),
            "heartrate": XMLField("globalmeasurements", "heartrate/#text", "str"),
            "rrint": XMLField("globalmeasurements", "rrint/#text", "str"),
            "print": XMLField("globalmeasurements", "print/#text", "str"),
            "qonset": XMLField("globalmeasurements", "qonset/#text", "str"),
            "qrsdur": XMLField("globalmeasurements", "qrsdur/#text", "str"),
            "qtint": XMLField("globalmeasurements", "qtint/#text", "str"),
            "qtcb": XMLField("globalmeasurements", "qtcb/#text", "str"),
            "qtcf": XMLField("globalmeasurements", "qtcf", "str"),
            "pfrontaxis": XMLField("globalmeasurements", "pfrontaxis/#text", "str"),
            "i40frontaxis": XMLField("globalmeasurements", "i40frontaxis/#text", "str"),
            "t40frontaxis": XMLField("globalmeasurements", "t40frontaxis/#text", "str"),
            "qrsfrontaxis": XMLField("globalmeasurements", "qrsfrontaxis/#text", "str"),
            "stfrontaxis": XMLField("globalmeasurements", "stfrontaxis/#text", "str"),
            "tfrontaxis": XMLField("globalmeasurements", "tfrontaxis/#text", "str"),
            "phorizaxis": XMLField("globalmeasurements", "phorizaxis/#text", "str"),
            "i40horizaxis": XMLField("globalmeasurements", "i40horizaxis/#text", "str"),
            "t40horizaxis": XMLField("globalmeasurements", "t40horizaxis/#text", "str"),
            "qrshorizaxis": XMLField("globalmeasurements", "qrshorizaxis/#text", "str"),
            "sthorizaxis": XMLField("globalmeasurements", "sthorizaxis/#text", "str"),
            "thorizaxis": XMLField("globalmeasurements", "thorizaxis/#text", "str"),
            "md_signature_line": XMLField("interpretation", "mdsignatureline", "str"),
            "severity_code": XMLField("severity", "@code", "str"),
            "severity": XMLField("severity", "#text", "str"),
        }
    }
}

cdef class SierraXMLECG(Parser):

    def __cinit__(self):
        pass

    @staticmethod
    cdef readonly object c_is_file_valid(str filepath):
        """

        @param filepath: 
        @return: 
        """
        if os.path.isfile(filepath) and filepath.lower().endswith(".xml"):
            try:
                xdom = minidom.parse(filepath)
                root = get_node(xdom, "restingecgdata")
                _, _ = assert_version(root)
                fd = open(filepath, "rb")
                doc = xmltodict.parse(fd.read())
                return fd, doc
            except Exception:
                pass
        return False

    @staticmethod
    def is_file_valid(filepath: str) -> bool:
        return SierraXMLECG.c_is_file_valid(filepath)

    cdef void read_patient_data(self, object doc, dict record) except*:
        """

        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            str field_name
            object field
        record["patient"] = {}
        for field_name, field in FIELDS["patient"].items():
            record["patient"][field_name] = field.get_value(doc)

    cdef void read_recording_data(self, object doc, dict record) except*:
        """

        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            dict recording_entry
            str combined_datetime
            int duration
        recording_entry = {}
        combined_datetime = f"{FIELDS['recording']['start_date'].get_value(doc)} {FIELDS['recording']['start_time'].get_value(doc)}"
        recording_entry["started_at"] = datetime.strptime(combined_datetime, '%Y-%m-%d %H:%M:%S')
        duration = FIELDS["recording"]["duration"].get_value(doc) // 1000
        recording_entry["ended_at"] = recording_entry["started_at"] + timedelta(seconds=duration)
        record["recording"] = recording_entry

    cdef void read_signal_characteristics(self, object doc, dict record) except*:
        """

        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            str field_name
            object field
            dict fields_entry
            dict entry
        record["signal_characteristics"] = []
        for fields_entry in FIELDS["signal_characteristics"]:
            entry = {}
            for field_name, field in fields_entry.items():
                if type(field) is XMLField:
                    entry[field_name] = field.get_value(doc)
                else:
                    entry[field_name] = field
            if entry["origin"] is None: continue
            record["signal_characteristics"].append(entry)

    cdef void read_signals_data(self, str filepath, dict record) except*:
        """

        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            object sierra_ecg
            list leads
            dict leads_record
            Py_ssize_t  i
        leads_record = {}
        sierra_ecg = read_file(filepath)
        leads = sierra_ecg.leads
        record["leads"] = {}
        for i in range(len(leads)):
            record["ecg"]["sampling_rate"] = leads[i].sampling_freq
            record["ecg"]["duration"] = leads[i].duration//1000
            record["leads"][f"lead_{leads[i].label}".lower()] = leads[i].samples.tolist()

    cdef void read_annotations_data(self, object doc, dict record) except*:
        """
        @param doc: 
        @param record: 
        @return: 
        """
        cdef:
            str field_name
            object field
            object statements
            dict statement
        record["annotations"] = {}
        record["annotations"]["expert"] = {}
        for field_name, field in FIELDS["annotations"]["expert"].items():
            record["annotations"]["expert"][field_name] = field.get_value(doc)
        statements = xmlutils.find_xml_dict_tag(doc, "statement")
        if type(statements) is list:
            for statement in statements:
                record["annotations"]["expert"][f"statement_{statement['statementcode']}_left"] = statement["leftstatement"]
                record["annotations"]["expert"][f"statement_{statement['statementcode']}_right"] = statement["rightstatement"]
        elif type(statements) is dict:
            record["annotations"]["expert"][f"statement_{statements['statementcode']}_left"] = statements["leftstatement"]
            record["annotations"]["expert"][f"statement_{statements['statementcode']}_right"] = statements[
                "rightstatement"]


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
        self.read_recording_data(doc, record)
        self.read_patient_data(doc, record)
        self.read_annotations_data(doc, record)
        self.read_signal_characteristics(doc, record)
        self.read_signals_data(filepath, record)
        record['ecg']['filepath'] = filepath
        fd.close()
        return record

