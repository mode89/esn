import math

class Noise :

    def __init__( self, persistence, octave_count ) :
        self._persistence = persistence
        self._octave_count = octave_count

    def __call__( self, x ) :
        retval = 0
        for octave in range( self._octave_count ) :
            frequency = 2 ** octave
            amplitude = self._persistence ** octave
            retval += self._interpolated_random( x * frequency ) * amplitude
        return retval

    def _interpolated_random( self, x ) :
        x_int = int( x )
        x_fract = x - x_int
        x1 = Noise._random( x_int )
        x2 = Noise._random( x_int + 1 )
        return Noise._cosine_interpolate( x1, x2, x_fract )

    @staticmethod
    def _random( x ) :
        x = ( x << 13 ) ^ x
        return 1.0 - ( ( x * ( x * x * 15731 + 789221 ) + 1376312589 ) &
            0x7fffffff ) / 1073741824.0

    @staticmethod
    def _cosine_interpolate( a, b, x ) :
        f = ( 1 - math.cos( x * math.pi ) ) * 0.5
        return a * ( 1 - f ) + b * f
