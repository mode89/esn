from math import *

class Gaussian :

    def __init__( self, a, b, c ) :
        self.a = a
        self.b = b
        self.c = c

    def __call__( self, x ) :
        return self.a * exp( -( ( x - self.b ) ** 2 ) /
            ( ( 2 * self.c ) ** 2 ) )

class GaussianPulse :

    def __init__( self, amplitude, width ) :
        self.gaussian = Gaussian( a = amplitude, b = ( width / 2 ),
            c = width / 10 )

    def __call__( self, x ) :
        return self.gaussian( x )
