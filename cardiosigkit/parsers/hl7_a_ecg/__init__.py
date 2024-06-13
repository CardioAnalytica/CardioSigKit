# -*- coding: utf-8 -*-
# Python Source File

from .hl7_a_ecg import HL7aECG

__registry__ = [
    {'extension': 'xml', 'format': 'Hl7 aECG', 'parser': 'HL7aECG'}
]