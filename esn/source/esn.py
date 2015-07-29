from ctypes import *
from ctypes.util import find_library
import inspect
import os

# Search for ESN shared library

_DLL_BASE_NAME = "esn"
if os.name == "nt" :
    _DLL_NAME = _DLL_BASE_NAME + ".dll"
else :
    _DLL_NAME = "lib" + _DLL_BASE_NAME + ".so"

FIND_LIBRARY_PATHS = [
    _DLL_BASE_NAME,
    os.path.join( os.getcwd(), _DLL_BASE_NAME )
]

frames = inspect.stack()
for frame in frames :
    frameDir = os.path.dirname( frame[1] )
    checkPath = os.path.join( frameDir, _DLL_BASE_NAME )
    FIND_LIBRARY_PATHS.append( checkPath )

for path in FIND_LIBRARY_PATHS :
    _DLL_PATH = find_library( path )
    if _DLL_PATH != None :
        break
    else :
        pathDir = os.path.dirname( path )
        checkPath = os.path.join( pathDir, _DLL_NAME )
        if os.path.exists( checkPath ) :
            _DLL_PATH = checkPath
            break

if _DLL_PATH == None :
    raise RuntimeError( "Failed to find ESN shared library." )

_DLL = cdll.LoadLibrary( _DLL_PATH )

class Network :

    def __init__( self, pointer ) :
        self.pointer = pointer

    def __del__( self ) :
        self.release()

    def release( self ) :
        _DLL.esnNetworkDestruct( self.pointer )

    def step( self, step ) :
        _DLL.esnNetworkStep( self.pointer, c_float( step ) )

    def capture_output( self, count ) :
        OutputArrayType = c_float * count
        outputArray = OutputArrayType()
        _DLL.esnNetworkCaptureOutput( self.pointer,
            pointer( outputArray ), count )
        output = [ outputArray[ i ] for i in range( count ) ]
        return output

    def train_online( self, output, forceOutput = False ) :
        OutputArrayType = c_float * len( output )
        outputArray = OutputArrayType( *output )
        _DLL.esnNetworkTrainOnline( self.pointer, pointer( outputArray ),
            len( output ), forceOutput )

class NetworkParamsNSLI( Structure ) :
    _fields_ = [
            ( "structSize", c_uint ),
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
        structSize = sizeof( NetworkParamsNSLI ),
        inputCount = inputCount,
        neuronCount = neuronCount,
        outputCount = outputCount,
        leakingRate = leakingRate,
        spectralRadius = spectralRadius,
        onlineTrainingForgettingFactor = onlineTrainingForgettingFactor,
        onlineTrainingInitialCovariance = onlineTrainingInitialCovariance )

    _DLL.esnCreateNetworkNSLI.restype = c_void_p
    return Network( _DLL.esnCreateNetworkNSLI( pointer( params ) ) )
