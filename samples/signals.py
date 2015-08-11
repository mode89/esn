from math import *
import random

class Gaussian :

    def __init__( self, a, b, c ) :
        self.a = a
        self.b = b
        self.c = c

    def __call__( self, x ) :
        return self.a * exp( -( ( x - self.b ) ** 2 ) /
            ( ( 2.0 * self.c ) ** 2 ) )

class GaussianPulse :

    def __init__( self, amplitude, width ) :
        self.gaussian = Gaussian( a = amplitude, b = ( width / 2.0 ),
            c = width / 10.0 )

    def __call__( self, x ) :
        return self.gaussian( x )

class PerlinNoise :

    def __init__( self, persistence, octave_count ) :
        self._persistence = persistence
        self._octave_count = octave_count
        self.shift = random.uniform(0.0, 1000000.0)

    def __call__( self, x ) :
        retval = 0
        for octave in range( self._octave_count ) :
            frequency = 2 ** octave
            amplitude = self._persistence ** octave
            retval += self._interpolated_random(
                ( x + self.shift ) * frequency ) * amplitude
        return retval

    def seed( self, value ) :
        self.shift = value * 1000000.0

    def _interpolated_random( self, x ) :
        x_int = int( x )
        x_fract = x - x_int
        x1 = PerlinNoise._random( x_int )
        x2 = PerlinNoise._random( x_int + 1 )
        return PerlinNoise._cosine_interpolate( x1, x2, x_fract )

    @staticmethod
    def _random( x ) :
        x = ( x << 13 ) ^ x
        return 1.0 - ( ( x * ( x * x * 15731 + 789221 ) + 1376312589 ) &
            0x7fffffff ) / 1073741824.0

    @staticmethod
    def _cosine_interpolate( a, b, x ) :
        f = ( 1 - cos( x * pi ) ) * 0.5
        return a * ( 1 - f ) + b * f
