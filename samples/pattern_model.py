import esn
import signals

SEED = 1
PATTERN_LENGTH = 1
PATTERN_PAUSE = 0.5

class Signal :

    def __init__( self ) :
        self.value = 0
        self.time = 0
        self.front_edge_time = 0
        self.pattern_noise = \
            signals.PerlinNoise( persistence=1, octave_count=7 )
        if SEED > 0 :
            self.pattern_noise.seed( SEED )
        self.pulse_noise = \
            signals.PerlinNoise( persistence=0.5, octave_count=1 )
        if SEED > 0 :
            self.pulse_noise.seed( SEED + 1 )
        self.prev_pulse_noise = self.pulse_noise( 0 )
        self.cur_pulse_noise = self.pulse_noise( 0 )

    def step( self, step ) :
        if self.time > ( self.front_edge_time + PATTERN_LENGTH + \
           PATTERN_PAUSE ) and self.is_front_edge() :
            self.front_edge_time = self.time

        if self.front_edge_time <= self.time and \
           self.time <= ( self.front_edge_time + PATTERN_LENGTH ) :
            self.value = self.pattern_noise( self.time - \
                    self.front_edge_time ) * 0.15
        else :
            self.value = 0

        self.prev_pulse_noise = \
            self.pulse_noise( self.time )
        self.time += step

    def is_front_edge( self ) :
        if self.prev_pulse_noise <= 0 and \
            self.pulse_noise( self.time ) > 0 :
            return True
        else :
            return False

class Model :

    def __init__( self, neuron_count ) :
        self.network = esn.CreateNetworkNSLI(
                inputCount=1,
                neuronCount=neuron_count,
                outputCount=1
            )
        self.noise = signals.PerlinNoise( persistence=0.5, octave_count=8 )
        if SEED > 0 :
            self.noise.seed( SEED + 2 )
        self.pattern = Signal()
        self.time = 0

    def step( self, step ) :
        self.pattern.step( step )
        self.noise_value = self.noise( self.time ) * 0.1
        self.input = self.noise_value + self.pattern.value
        self.network.set_inputs( [ self.input ] )
        self.network.step( step )
        self.output = self.network.capture_output( 1 )[ 0 ]

        print( "%10s %10s %10s %10s" %
                (
                    str( "%0.3f" % self.time ),
                    str( "%0.5f" % self.input ),
                    str( "%0.5f" % self.pattern.value ),
                    str( "%0.5f" % self.output )
                )
            )

        self.time += step
