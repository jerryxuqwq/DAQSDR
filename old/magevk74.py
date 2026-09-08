import unittest
import itertools
import ctypes
import argparse
import sys
import importlib.util
from os.path import join

def DefaultModulePath():
    if sys.platform == 'win32':
        modulePath = join('C:/', 'Program Files', 'Vayyar', 'vtrigU', 'python', 'vtrigU.py')
    elif sys.platform.startswith('linux'):
        modulePath = join('/usr', 'share', 'vtrigU', 'python', 'vtrigU.py')
    else:
        raise BaseException('Unsupported platform: ' + sys.platform)
    return modulePath

def Import_vtrigU():
    global vtrig
    modulePath = DefaultModulePath()
    spec = importlib.util.spec_from_file_location('vtrigU', modulePath)
    vtrig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vtrig)

if __name__ == '__main__':
    Import_vtrigU()

    vtrig.Init() 
    # apply settings:
    vtrigSettings = vtrig.RecordingSettings(
        vtrig.FrequencyRange(65.0*1000, 65.5*1000, 100), # 101 points, from 65.0-66.0 GHz
        30.0, # RBW (in KHz)
        vtrig.VTRIG_U_TXMODE__HIGH_RATE #
        ) 
    vtrig.ApplySettings(vtrigSettings)

    vtrig.Record() # one recording
    
    # modify settings
    vtrigSettings.rbw_khz = 30.5
    vtrigSettings.mode = vtrig.VTRIG_U_TXMODE__MED_RATE
    vtrig.ApplySettings(vtrigSettings)
    
    # record a bunch of times
    #for i in range(10):
    while True:
        vtrig.Record()  
    
    actual_freqs = vtrig.GetFreqVector_MHz()
    pair_list = vtrig.GetAntennaPairs(vtrigSettings.mode)

    print("FREQUENCIES: ")
    print(actual_freqs)
    print("------")
    recording = vtrig.GetRecordingResult()
    for pair_id in range(len(pair_list)):
        curPair = pair_list[pair_id]
        print(str(curPair) + ":" + str(recording[curPair]))
