import esn
import perlin

class Model :

    def __init__( self, neuron_count ) :
        self._network = esn.CreateNetworkNSLI(
                inputCount=1,
                neuronCount=neuron_count,
                outputCount=1
            )
        self._noise = perlin.Noise( persistence=0.5, octave_count=8 )
        self._time = 0

    def step( self, step ) :
        self.input = self._noise( self._time )
        self._network.set_inputs( [ self.input ] )
        self._network.step( step )
        self.output = self._network.capture_output( 1 )[ 0 ]

        print( "%10s %10s %10s" %
                (
                    str( "%0.3f" % self._time ),
                    str( "%0.5f" % self.input ),
                    str( "%0.5f" % self.output )
                )
            )

        self._time += step
