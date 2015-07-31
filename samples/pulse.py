from math import *
import esn
import imp
import random

NEURON_COUNT = 100
LEAKING_RATE = 0.1
CONNECTIVITY = 0.1
SIM_STEP = 0.01
INTERVAL_WIDTH_MIN = 0.1
INTERVAL_WIDTH_MAX = 1.0
TARGET_INTERVAL_WIDTH = 0.7
TARGET_INTERVAL_ERROR = 0.1
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

model = Model()

frame = 0
while True :
    frame += 1
    model.simulate( frame )
