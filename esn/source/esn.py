from ctypes import *
from ctypes.util import find_library
from enum import Enum
import inspect
import os

_DLL_PATH = @ESN_PY_DLL_PATH@
# Need to check the path, otherwise CDLL.LoadLibrary()
# raises an exception under Windows.
if _DLL_PATH :
    _DLL = cdll.LoadLibrary( _DLL_PATH )

class Error( Enum ) :
    NO_ERROR = 0
    OUTPUT_IS_NOT_FINITE = 1

class OutputIsNotFinite( RuntimeError ) :
    def __init__( self ) :
        RuntimeError.__init__( self, "One or more outputs "
            "of the network are not finite values." )

def raise_on_error( code ) :
    if Error( code ) != Error.NO_ERROR :
        raise {
                Error.OUTPUT_IS_NOT_FINITE : OutputIsNotFinite()
            }[ Error( code ) ]

class Network :

    def __init__( self, pointer ) :
        self.pointer = pointer

    def __del__( self ) :
        self.release()

    def release( self ) :
        _DLL.esnNetworkDestruct( self.pointer )

    def set_inputs( self, inputs ) :
        InputsArrayType = c_float * len( inputs )
        inputsArray = InputsArrayType( *inputs )
        _DLL.esnNetworkSetInputs( self.pointer, pointer( inputsArray ),
            len( inputs ) )

    def set_input_scalings( self, scalings ) :
        ScalingsArrayType = c_float * len( scalings )
        scalingsArray = ScalingsArrayType( *scalings )
        _DLL.esnNetworkSetInputScalings( self.pointer,
            pointer( scalingsArray ), len( scalings ) )

    def set_input_bias( self, bias ) :
        BiasArrayType = c_float * len( bias )
        biasArray = BiasArrayType( *bias )
        _DLL.esnNetworkSetInputBias( self.pointer,
            pointer( biasArray ), len( bias ) )

    def set_feedback_scalings( self, scalings ) :
        ScalingsArrayType = c_float * len( scalings )
        scalingsArray = ScalingsArrayType( *scalings )
        _DLL.esnNetworkSetFeedbackScalings( self.pointer,
            pointer( scalingsArray ), len( scalings ) )

    def step( self, step ) :
        retval = _DLL.esnNetworkStep( self.pointer, c_float( step ) )
        raise_on_error( retval )

    def capture_transformed_inputs( self, count ) :
        InputArrayType = c_float * count
        inputArray = InputArrayType()
        _DLL.esnNetworkCaptureTransformedInput( self.pointer,
            pointer( inputArray ), count )
        return [ inputArray[i] for i in range( count ) ]

    def capture_activations( self, count ) :
        ActivationsArrayType = c_float * count
        activationsArray = ActivationsArrayType()
        _DLL.esnNetworkCaptureActivations( self.pointer,
            pointer( activationsArray ), count )
        return [ activationsArray[i] for i in range( count ) ]

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
            ( "leakingRateMin", c_float ),
            ( "leakingRateMax", c_float ),
            ( "useOrthonormalMatrix", c_bool ),
            ( "spectralRadius", c_float ),
            ( "connectivity", c_float ),
            ( "linearOutput", c_bool ),
            ( "onlineTrainingForgettingFactor", c_float ),
            ( "onlineTrainingInitialCovariance", c_float )
        ]

def create_network(
    inputCount,
    neuronCount,
    outputCount,
    leakingRateMin = 0.1,
    leakingRateMax = 1.0,
    useOrthonormalMatrix = True,
    spectralRadius = 1.0,
    connectivity = 1.0,
    linearOutput = False,
    onlineTrainingForgettingFactor = 1.0,
    onlineTrainingInitialCovariance = 1000.0 ) :

    if not _DLL._name :
        raise RuntimeError( "ESN shared library hasn't been loaded." )

    params = NetworkParamsNSLI(
        structSize = sizeof( NetworkParamsNSLI ),
        inputCount = inputCount,
        neuronCount = neuronCount,
        outputCount = outputCount,
        leakingRateMin = leakingRateMin,
        leakingRateMax = leakingRateMax,
        useOrthonormalMatrix = useOrthonormalMatrix,
        spectralRadius = spectralRadius,
        connectivity = connectivity,
        linearOutput = linearOutput,
        onlineTrainingForgettingFactor = onlineTrainingForgettingFactor,
        onlineTrainingInitialCovariance = onlineTrainingInitialCovariance )

    _DLL.esnCreateNetworkNSLI.restype = c_void_p
    return Network( _DLL.esnCreateNetworkNSLI( pointer( params ) ) )
