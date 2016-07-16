%module esn

%{
#include "esn/network.hpp"
#include "esn/network_nsli.hpp"
%}

%include <std_shared_ptr.i>
%shared_ptr(ESN::Network)

%include "esn/export.h"
%include "esn/network.hpp"
%include "esn/network_nsli.hpp"
