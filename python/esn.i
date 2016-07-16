%module esn

%{
#include "esn/network.h"
#include "esn/network_nsli.h"
%}

%include <std_shared_ptr.i>
%shared_ptr(ESN::Network)

%include <std_vector.i>
%template(Vector) std::vector<float>;

%include "esn/export.h"
%include "esn/network.h"
%include "esn/network_nsli.h"
