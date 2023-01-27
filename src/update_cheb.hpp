#ifndef UPDATE_CHEB_HPP
#define UPDATE_CHEB_HPP

#include "static_vars.hpp"
#include<complex>




void update_cheb ( type*, type*, type*, type*, type , type );
void update_cheb ( int ,  type*, type*, type*, type*, type*, type , type );
void vel_op (type*, type*);
void batch_vel_op (type*, type*);

#endif //UPDATE_CHEB_HPP
