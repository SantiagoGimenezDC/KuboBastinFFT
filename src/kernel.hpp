#ifndef KERNEL_HPP
#define KERNEL_HPP


#include"static_vars.hpp"
#include<cmath>

inline type kernel(const int m){
  return 1/(M_+1.0)*
         (
 	   (M_-m+1.0)*
           cos(M_PI*m / (M_+1.0)) +
           sin(M_PI*m / (M_+1.0)) /
           tan(M_PI   / (M_+1.0))
	 );
}

#endif //KERNEL_HPP
