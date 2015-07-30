from math import *
import esn
import imp

# Check if we can use matplotlib
try :
    imp.find_module( "matplotlib" )
    from matplotlib import pyplot
    from matplotlib import animation
    USE_MATPLOTLIB = True
except ImportError :
    USE_MATPLOTLIB = False

NEURON_COUNT = 10
LEAKING_RATE = 0.2
CONNECTIVITY = 0.1
SINE_FREQ = 50.0
SINE_AMPLITUDE = 0.5
SIM_STEP = 0.001
TRAIN_TIME = 1 / SINE_FREQ
STEPS_PER_FRAME = 20

network = esn.CreateNetworkNSLI(
    inputCount = 1,
    neuronCount = NEURON_COUNT,
    outputCount  = 1,
    leakingRate = LEAKING_RATE,
    connectivity = CONNECTIVITY )

def simulate( frame ) :

    time = frame * SIM_STEP

    sine = sin( 2 * pi * SINE_FREQ * time ) * SINE_AMPLITUDE

    network.step( SIM_STEP )
    output = network.capture_output( 1 )

    if time < TRAIN_TIME :
        network.train_online( [ sine ], True )

    print( "%10s %10s %10s" %
            (
                str( "%0.5f" % time ),
                str( "%0.5f" % sine ),
                str( "%0.5f" % output[0] )
            )
        )

    return [ time, sine, output[0] ]

if USE_MATPLOTLIB :

    figure = pyplot.figure()
    subplot = figure.add_subplot( 111 )
    sineLine, outputLine = subplot.plot( [], [], [], [] )
    subplot.set_xlim( -1, 1 )
    subplot.set_ylim( -1.1 * SINE_AMPLITUDE, 1.1 * SINE_AMPLITUDE )
    subplot.grid( True )

    timeData = []
    sineData = []
    outputData = []

    def animationFunc( frame ) :
        for i in range( 0, STEPS_PER_FRAME ) :
            time, sine, output = simulate( i + STEPS_PER_FRAME * frame )
            timeData.append( time )
            sineData.append( sine )
            outputData.append( output )
        sineLine.set_data( timeData, sineData )
        outputLine.set_data( timeData, outputData )
        subplot.set_xlim( time - 1 / SINE_FREQ, time + 0.1 / SINE_FREQ )

    anim = animation.FuncAnimation( figure, animationFunc, interval = 30 )

    pyplot.show()

else :

    frame = 0
    while True :
        frame += 1
        simulate( frame )
