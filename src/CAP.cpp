#include<cmath>
#include<iostream>

#include "static_vars.hpp"
#include"CAP.hpp"



void CAP::create_CAP(int W, int C, int LE, type dmp_op[]){
  type func, y, c=2.66206,
    gamma, gamma_eta=exp(-asinh(-eta_));

  
  int DIM = LE*W,
    SUBDIM = (2*C+LE)*W;

 int ctLe  = C*W;

  for(int i=0;i<SUBDIM;i++)
    dmp_op[i+C*W] = gamma_eta;

  for(int i=0;i<C*W;i++){
    dmp_op[i] = 1.0;
    dmp_op[DIM-i] = 1.0;    
  }


  
  for(int i=0;i<ctLe;i++){
    if(C!=0){
      y     = (type)(i/W)/C; 
      func  = (-4.0*Emin_/(c*c))*(1/pow(1-y,2)+1/pow(1+y,2)-2)+eta_;
      gamma = asinh(-func);
      gamma = exp(-gamma);
    }
    else
      gamma = gamma_eta;

    dmp_op[ctLe-i-1]       *= gamma; 
    dmp_op[DIM-ctLe+i]     *= gamma;
      
  }
}


void create_CAP(int W, int C, int LE, type eta, type Emin, type dmp_op[]){
  type func, y, c=2.66206,
    gamma, gamma_eta=exp(-asinh(-eta));

  
  int SUBDIM = LE*W,
    DIM = (2*C+LE)*W;

 int ctLe  = C*W;

  for(int i=0;i<SUBDIM;i++)
    dmp_op[i+C*W] = gamma_eta;

  for(int i=0;i<C*W;i++){
    dmp_op[i] = 1.0;
    dmp_op[DIM-i] = 1.0;    
  }


  
  for(int i=0;i<ctLe;i++){
    if(C!=0){
      y     = (type)(i/W)/C; 
      func  = (-4.0*Emin/(c*c))*(1/pow(1-y,2)+1/pow(1+y,2)-2)+eta;
      gamma = asinh(-func);
      gamma = exp(-gamma);
    }
    else
      gamma = gamma_eta;

    dmp_op[ctLe-i-1]       *= gamma; 
    dmp_op[DIM-ctLe+i]     *= gamma;
      
  }
}
