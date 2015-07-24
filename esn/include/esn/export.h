#ifndef __ESN_EXPORT_H__
#define __ESN_EXPORT_H__

#ifdef WIN32
    #ifdef esn_EXPORTS
        #define ESN_EXPORT __declspec( dllexport )
    #else
        #define ESN_EXPORT __declspec( dllimport )
    #endif
#else
    #define ESN_EXPORT
#endif

#endif // __ESN_EXPORT_H__
