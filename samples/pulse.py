from math import *
import imp
import random

esn = imp.load_source( "esn", "@ESN_PYTHON_MODULE@" )
esn.load_library( "@ESN_DLL@" )

# Check if we can use matplotlib
try :
    imp.find_module( "matplotlib" )
    from matplotlib import pyplot
    from matplotlib import animation
    USE_MATPLOTLIB = True
except ImportError :
    USE_MATPLOTLIB = False

NEURON_COUNT = 100
LEAKING_RATE = 0.1
CONNECTIVITY = 0.1
SIM_STEP = 0.01
INTERVAL_WIDTH_MIN = 0.1
INTERVAL_WIDTH_MAX = 1.0
TARGET_INTERVAL_WIDTH = 0.7
TARGET_INTERVAL_ERROR = 0.1
INPUT_PULSE_MAX = 1.0
OUTPUT_PULSE_WIDTH = 0.1
OUTPUT_PULSE_MAX = 1.0
OUTPUT_PULSE_THRESHOLD = 0.0001
TRAIN_PULSE_COUNT = 100
STEPS_PER_FRAME = 20

class Model :

    def __init__( self ) :

        self.network = esn.CreateNetworkNSLI(
            inputCount = 1,
            neuronCount = NEURON_COUNT,
            outputCount  = 1,
            leakingRate = LEAKING_RATE,
            connectivity = CONNECTIVITY
        )

        self.inputState = False
        self.inputs = [ 0 ]
        self.currentIntervalWidth = 0
        self.nextInterval = 0
        self.outputPulseStart = 0
        self.outputPulseAmplitude = 0
        self.outputPulseCount = 0

    def pulse( self, x, width, amplitude ) :

        retval = amplitude * exp( -pow( 6 * ( x - width / 2 ) / width, 2 ) )
        return 0 if retval < OUTPUT_PULSE_THRESHOLD else retval

    def simulate( self, frame ) :

        time = frame * SIM_STEP

        if time > self.nextInterval :
            error = abs( self.currentIntervalWidth - TARGET_INTERVAL_WIDTH )
            if self.inputState and error < TARGET_INTERVAL_ERROR :
                self.outputPulseStart = time
                self.outputPulseAmplitude = ( TARGET_INTERVAL_ERROR -   \
                    abs( self.currentIntervalWidth -                    \
                        TARGET_INTERVAL_WIDTH ) ) /                     \
                    TARGET_INTERVAL_ERROR * OUTPUT_PULSE_MAX
                self.outputPulseCount += 1

            self.inputState = not self.inputState
            self.inputs[0] = 1 if self.inputState else 0

            self.currentIntervalWidth = random.uniform( INTERVAL_WIDTH_MIN,
                INTERVAL_WIDTH_MAX )
            self.nextInterval += self.currentIntervalWidth

        self.inputs[0] = self.pulse( self.nextInterval - time,
            self.currentIntervalWidth, INPUT_PULSE_MAX ) if self.inputState else 0

        referenceOutput = self.pulse( time - self.outputPulseStart,
            OUTPUT_PULSE_WIDTH, self.outputPulseAmplitude );

        self.network.set_inputs( self.inputs )
        self.network.step( SIM_STEP )
        output = self.network.capture_output( 1 )

        if self.outputPulseCount < TRAIN_PULSE_COUNT :
            self.network.train_online( [ referenceOutput ], True );

        print( "%10s %7s %10s %10s" %
                (
                    str( "%0.5f" % time ),
                    str( "%r" % self.inputState ),
                    str( "%0.5f" % referenceOutput ),
                    str( "%0.5f" % output[0] )
                )
            )

        return time, self.inputs[0], referenceOutput, output[0]

model = Model()

if USE_MATPLOTLIB :

    figure = pyplot.figure()
    subplot = figure.add_subplot( 111 )
    inputLine, refLine, outputLine = subplot.plot( [], [], [], [], [], [] )
    subplot.set_ylim( -0.1, 1.1 )
    subplot.grid( True )

    timeData = []
    inputData = []
    refData = []
    outputData = []

    def animationFunc( frame ) :
        for i in range( 0, STEPS_PER_FRAME ) :
            time, input, ref, output = \
                model.simulate( i + STEPS_PER_FRAME * frame )
            timeData.append( time )
            inputData.append( input )
            refData.append( ref )
            outputData.append( output )
        inputLine.set_data( timeData, inputData )
        refLine.set_data( timeData, refData )
        outputLine.set_data( timeData, outputData )
        subplot.set_xlim( time - 1, time + 0.1 )

    anim = animation.FuncAnimation( figure, animationFunc, interval = 30 )

    pyplot.show()

else :

    frame = 0
    while True :
        frame += 1
        model.simulate( frame )
