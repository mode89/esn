import esn
import perlin

PATTERN_LENGTH = 1
PATTERN_PAUSE = 0.5

class Signal :

    def __init__( self, seed ) :
        self.seed = seed
        self.value = 0
        self.time = 0
        self.front_edge_time = 0
        self.pattern_noise = \
            perlin.Noise( persistence=0.5, octave_count=8 )
        self.pulse_noise = \
            perlin.Noise( persistence=0.5, octave_count=1 )
        self.prev_pulse_noise = self.pulse_noise( 0 )
        self.cur_pulse_noise = self.pulse_noise( 0 )

    def step( self, step ) :
        if self.time > ( self.front_edge_time + PATTERN_LENGTH + \
           PATTERN_PAUSE ) and self.is_front_edge() :
            self.front_edge_time = self.time

        if self.front_edge_time <= self.time and \
           self.time <= ( self.front_edge_time + PATTERN_LENGTH ) :
            self.value = self.pattern_noise( self.time - \
                    self.front_edge_time + self.seed ) * 0.3 + 0.5
        else :
            self.value = 0

        self.prev_pulse_noise = \
            self.pulse_noise( self.time + self.seed )
        self.time += step

    def is_front_edge( self ) :
        if self.prev_pulse_noise <= 0 and \
            self.pulse_noise( self.time + self.seed ) > 0 :
            return True
        else :
            return False

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
