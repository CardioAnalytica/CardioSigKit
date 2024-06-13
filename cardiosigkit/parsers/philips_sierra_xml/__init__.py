# -*- coding: utf-8 -*-
# Python Source File

from .sierra_xml import SierraXMLECG

__registry__ = [
    {'extension': 'xml', 'format': 'Phillips Sierra ECG XML', 'parser': 'SierraXMLECG'}
]