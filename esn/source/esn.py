from ctypes import *
from ctypes.util import find_library
import inspect
import os

# Search for ESN shared library

__LIB_BASE_NAME = "esn"
if os.name == "nt" :
    __LIB_NAME = __LIB_BASE_NAME + ".dll"
else :
    __LIB_NAME = "lib" + __LIB_BASE_NAME + ".so"

FIND_LIBRARY_PATHS = [
    __LIB_BASE_NAME,
    os.path.join( os.getcwd(), __LIB_BASE_NAME )
]

frames = inspect.stack()
for frame in frames :
    frameDir = os.path.dirname( frame[1] )
    checkPath = os.path.join( frameDir, __LIB_BASE_NAME )
    FIND_LIBRARY_PATHS.append( checkPath )

for path in FIND_LIBRARY_PATHS :
    __LIB_PATH = find_library( path )
    if __LIB_PATH != None :
        break
    else :
        pathDir = os.path.dirname( path )
        checkPath = os.path.join( pathDir, __LIB_NAME )
        if os.path.exists( checkPath ) :
            __LIB_PATH = checkPath
            break

if __LIB_PATH == None :
    raise RuntimeError( "Failed to find ESN shared library." )

__LIB = cdll.LoadLibrary( __LIB_PATH )

class NetworkParamsNSLI( Structure ) :
    _fields_ = [
            ( "inputCount", c_uint ),
            ( "neuronCount", c_uint ),
            ( "outputCount", c_uint ),
            ( "leakingRate", c_float ),
            ( "spectralRadius", c_float ),
            ( "onlineTrainingForgettingFactor", c_float ),
            ( "onlineTrainingInitialCovariance", c_float )
        ]

def CreateNetworkNSLI(
    inputCount,
    neuronCount,
    outputCount,
    leakingRate = 1.0,
    spectralRadius = 1.0,
    onlineTrainingForgettingFactor = 0.999,
    onlineTrainingInitialCovariance = 1000.0 ) :

    params = NetworkParamsNSLI(
        inputCount = inputCount,
        neuronCount = neuronCount,
        outputCount = outputCount,
        leakingRate = leakingRate,
        spectralRadius = spectralRadius,
        onlineTrainingForgettingFactor = onlineTrainingForgettingFactor,
        onlineTrainingInitialCovariance = onlineTrainingInitialCovariance )

    func = __LIB.esnCreateNetworkNSLI
    func.restype = c_void_p

    return func( pointer( params ) )
