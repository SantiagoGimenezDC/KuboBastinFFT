#ifndef POL_CYCLE_HPP
#define POL_CYCLE_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<cstring>

#include <sys/stat.h>
#include <sys/types.h>



#include "static_vars.hpp"
#include "update_cheb.hpp"
#include "kernel.hpp"

void polynomial_cycle(type polys[M_*SUBDIM_], type vec[DIM_], type p_vec[DIM_], type pp_vec[DIM_], type damp_op[DIM_],  type a, type b){
   
//=================================KPM Step 0======================================//


  for(int i=0;i<SUBDIM_;i++)
    polys[i*M_] = p_vec[i+C_*W_];
    


  
//=================================KPM Step 1======================================//   
    
    
    update_cheb ( 1, polys, vec, p_vec, pp_vec, damp_op, 2*a, b);
    


//=================================KPM Steps 2 and on===============================//
    
  for( int m=2; m<M_; m++ )
    update_cheb( m, polys, vec, p_vec, pp_vec, damp_op, a, b);
  

  
}





#endif //POL_CYCLE_HPP
