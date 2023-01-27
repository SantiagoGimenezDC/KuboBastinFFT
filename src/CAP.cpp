#include<cmath>
#include "static_vars.hpp"

#include<iostream>
#include"CAP.hpp"


void set_CAP(type Emin,  type dmp_op[DIM_]){
  type func, y, c=2.66206,
       gamma;
  
  int ctLe  = SPINS_*C_*W_;

  for(int i=0;i<DIM_;i++)
    dmp_op[i] = 1.0;
  
  for(int i=0;i<ctLe;i++){      
    y     = (type)(i/W_)/C_; 
    func  = (-4.0*Emin/(c*c))*(1/pow(1-y,2)+1/pow(1+y,2)-2);
    gamma = asinh(-func);
    gamma = exp(-gamma);


      if(SPINS_==1){
        dmp_op[ctLe-i-1]       *= gamma; 
        dmp_op[DIM_-ctLe+i]     *= gamma;
      }

      
      else {
        dmp_op[ctLe-2*i-1]     *= gamma;
        dmp_op[ctLe-2*i]       *= gamma;
	
        dmp_op[DIM_-ctLe+2*i]   *= gamma;
        dmp_op[DIM_-ctLe+1+2*i] *= gamma;
      }


      
    }
  }
