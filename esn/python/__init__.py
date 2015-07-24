from ctypes import *
from ctypes.util import find_library
import inspect
import os

# Search for ESN shared library

FIND_LIBRARY_PATHS = [
    "esn",
    os.path.join( os.getcwd(), "esn" )
]

frames = inspect.stack()
for frame in frames :
    frameDir = os.path.dirname( frame[1] )
    checkPath = os.path.join( frameDir, "esn" )
    FIND_LIBRARY_PATHS.append( checkPath )

for path in FIND_LIBRARY_PATHS :
    ESN_LIB_PATH = find_library( path )
    if ESN_LIB_PATH != None :
        break

if ESN_LIB_PATH == None :
    raise RuntimeError( "Failed to find ESN shared library." )

ESN_LIB = cdll.LoadLibrary( ESN_LIB_PATH )
