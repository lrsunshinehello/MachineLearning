#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:41:26 2018

@author: li
"""
from numpy import *
def loadSimpleData():
    datMat = mat([[1.,2.1],
                     [2.,1.1],
                     [1.3,1.],
                     [1.,1.],
                     [2.,1.]])
    classLables = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLables