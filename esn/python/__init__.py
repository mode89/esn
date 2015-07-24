from ctypes import *
from ctypes.util import find_library
import inspect
import os

# Search for ESN shared library

ESN_LIB_BASE_NAME = "esn"
if os.name == "nt" :
    ESN_LIB_NAME = ESN_LIB_BASE_NAME + ".dll"
else :
    ESN_LIB_NAME = "lib" + ESN_LIB_BASE_NAME + ".so"

FIND_LIBRARY_PATHS = [
    ESN_LIB_BASE_NAME,
    os.path.join( os.getcwd(), ESN_LIB_BASE_NAME )
]

frames = inspect.stack()
for frame in frames :
    frameDir = os.path.dirname( frame[1] )
    checkPath = os.path.join( frameDir, ESN_LIB_BASE_NAME )
    FIND_LIBRARY_PATHS.append( checkPath )

for path in FIND_LIBRARY_PATHS :
    ESN_LIB_PATH = find_library( path )
    if ESN_LIB_PATH != None :
        break
    else :
        pathDir = os.path.dirname( path )
        checkPath = os.path.join( pathDir, ESN_LIB_NAME )
        if os.path.exists( checkPath ) :
            ESN_LIB_PATH = checkPath
            break

if ESN_LIB_PATH == None :
    raise RuntimeError( "Failed to find ESN shared library." )

ESN_LIB = cdll.LoadLibrary( ESN_LIB_PATH )
