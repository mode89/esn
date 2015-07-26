from math import *
import esn

NEURON_COUNT = 100
LEAKING_RATE = 0.1
SINE_FREQ = 1.0
SIM_STEP = 0.01
TRAIN_TIME = 25.0 / SINE_FREQ

network = esn.CreateNetworkNSLI( inputCount = 1, neuronCount = NEURON_COUNT,
    outputCount  = 1, leakingRate = LEAKING_RATE )

def simulate( frame ) :

    time = frame * SIM_STEP

    sine = sin( 2 * pi * SINE_FREQ * time )

    network.step( SIM_STEP )
    output = network.capture_output( 1 )

    if time < TRAIN_TIME :
        network.train_online( [ sine ], True )

    return [ time, sine, output[0] ]

frame = 0
while True :

    frame += 1

    time, sine, output = simulate( frame )

    print( "%10s %10s %10s" %
            (
                str( "%0.5f" % time ),
                str( "%0.5f" % sine ),
                str( "%0.5f" % output )
            )
        )
