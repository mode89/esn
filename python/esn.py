from ctypes import *
from ctypes.util import find_library
from enum import Enum
import inspect
import os

_DLL_PATH = find_library( "esn" )
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

class NetworkParams(Structure) :
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
            ( "onlineTrainingInitialCovariance", c_float ),
            ( "hasOutputFeedback", c_bool )
        ]

class Network :

    def __init__(self,
        ins,
        outs,
        neurons,
        leak_min = 0.1,
        leak_max = 1.0,
        use_orth_mat = True,
        spect_rad = 1.0,
        cnctvty = 1.0,
        lin_out = False,
        has_ofb = True,
        forgetting = 1.0,
        covariance = 1000.0):
        if not _DLL._name :
            raise RuntimeError("ESN shared library hasn't been loaded.")

        params = NetworkParams(
            structSize=sizeof(NetworkParams),
            inputCount=ins,
            neuronCount=neurons,
            outputCount=outs,
            leakingRateMin=leak_min,
            leakingRateMax=leak_max,
            useOrthonormalMatrix=use_orth_mat,
            spectralRadius=spect_rad,
            connectivity=cnctvty,
            linearOutput=lin_out,
            hasOutputFeedback=has_ofb,
            onlineTrainingForgettingFactor=forgetting,
            onlineTrainingInitialCovariance=covariance)

        _DLL.esnCreateNetworkNSLI.restype = c_void_p
        self.pointer = _DLL.esnCreateNetworkNSLI(pointer(params))

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
