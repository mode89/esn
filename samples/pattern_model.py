import esn
import imp
import signals
import random

SEED = 0
PATTERN_LENGTH = 1
PATTERN_PAUSE = 0.5
OUTPUT_PULSE_AMPLITUDE = 0.9
OUTPUT_PULSE_LENGTH = 0.1
WASHOUT_TIME = 10.0
TRAIN_TIME = 100.0
VARIABLE_MAGNITUDE = True
FALSE_PATTERN = True
CONNECTIVITY = 0.5
TEACHER_FORCING = False
TRAINING_STRATEGY = "discontinuous"

class Signal :

    def __init__( self ) :
        self.magnitude = 1.0
        self.value = 0
        self.time = 0
        self.front_edge = 0
        self.back_edge = -OUTPUT_PULSE_LENGTH
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
        if self.time > ( self.front_edge + PATTERN_LENGTH + \
           PATTERN_PAUSE ) and self.is_front_edge() :
            self.front_edge = self.time
            self.back_edge = self.front_edge + PATTERN_LENGTH
            if VARIABLE_MAGNITUDE :
                self.magnitude = random.uniform( 0.3, 1.0 )

        if self.front_edge <= self.time and \
           self.time <= self.back_edge :
            self.value = self.pattern_noise( self.time - \
                    self.front_edge ) * 0.15 * self.magnitude
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
        params = esn.NetworkParams()
        params.inputCount = 1
        params.neuronCount = neuron_count
        params.outputCount = 1
        params.connectivity = CONNECTIVITY
        self.network = esn.CreateNetwork(params)
        trainerParams = esn.TrainerParams()
        self.trainer = esn.CreateTrainer(trainerParams, self.network)
        self.noise = signals.PerlinNoise( persistence=0.5, octave_count=8 )
        if SEED > 0 :
            self.noise.seed( SEED + 2 )
        self.pattern = Signal()
        if FALSE_PATTERN :
            self.false_pattern = Signal()
        self.train_pulse = signals.GaussianPulse(
            amplitude=OUTPUT_PULSE_AMPLITUDE,
            width=OUTPUT_PULSE_LENGTH )
        self.time = 0

    def step( self, step ) :
        self.pattern.step( step )
        self.noise_value = self.noise( self.time ) * 0.1
        self.input = self.noise_value + self.pattern.value
        if FALSE_PATTERN :
            self.false_pattern.step( step )
            self.input += self.false_pattern.value
        self.network.SetInputs([self.input])
        self.network.Step(step)
        outputs = esn.Vector(1)
        self.network.CaptureOutput(outputs)
        self.output = outputs[0]
        self.train_output = self.train_pulse( self.time - \
            self.pattern.back_edge )
        if self.time > WASHOUT_TIME and self.time < TRAIN_TIME :
            getattr( Model.TrainingStrategy, TRAINING_STRATEGY )( self )

        print( "%10s %10s %10s %10s %10s" %
                (
                    str( "%0.3f" % self.time ),
                    str( "%0.5f" % self.input ),
                    str( "%0.5f" % self.pattern.value ),
                    str( "%0.5f" % self.train_output ),
                    str( "%0.5f" % self.output )
                )
            )

        self.time += step

    class TrainingStrategy :

        @staticmethod
        def continuous( model ) :
            model.trainer.TrainOnline(
                [model.train_output], TEACHER_FORCING)

        @staticmethod
        def discontinuous( model ) :
            if model.time > model.pattern.back_edge and \
                model.time < ( model.pattern.back_edge + \
                    OUTPUT_PULSE_LENGTH ) :
                model.trainer.TrainOnline(
                    [model.train_output], TEACHER_FORCING)
            elif model.output > 0.3 or model.output < -0.3:
                model.trainer.TrainOnline(
                    [model.train_output], TEACHER_FORCING)
