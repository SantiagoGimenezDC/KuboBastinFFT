#include<cmath>
#include<iostream>
#include<omp.h>
#include<thread>

#include "static_vars.hpp"
#include"CAP.hpp"


void Mandelshtam::create_CAP(int W, int C, int LE, r_type dmp_op[]){
  r_type Emin=this->Emin(),
    eta=this->eta(),
    c=2.66206,
    gamma_eta=exp(-asinh(-eta));

  
  int SUBDIM = LE*W,
    DIM = (2*C+LE)*W;

 int ctLe  = C*W;

#pragma omp parallel for
  for(int i=0;i<SUBDIM;i++)
    dmp_op[i+C*W] = gamma_eta;

#pragma omp parallel for  
  for(int i=0;i<C*W;i++){
    dmp_op[i] = 1.0;
    dmp_op[DIM-i] = 1.0;    
  }


#pragma omp parallel for  
  for(int i=0;i<ctLe;i++){
    r_type gamma;
    if(C!=0){
      r_type y     = (r_type)(i/W)/C; 
      r_type func  = (-4.0*Emin/(c*c))*(1/pow(1-y,2)+1/pow(1+y,2)-2)+eta;
      gamma = asinh(-func);
      gamma = exp(-gamma);
    }
    else
      gamma = gamma_eta;

    dmp_op[ctLe-i-1]       *= gamma; 
    dmp_op[DIM-ctLe+i]     *= gamma;      
  }
}


void Effective_Contact::create_CAP(int W, int C, int LE, r_type dmp_op[]){

  r_type
    eta=this->eta(),
    gamma_eta = exp(-asinh(-eta));

  
  int DIM = (2*C+LE)*W;


#pragma omp parallel for  
  for(int i=0;i<DIM;i++)
    dmp_op[i] = 1.0;
    
  

#pragma omp parallel for  
  for(int i=0;i<W;i++){
    dmp_op[i]       *= gamma_eta; 
    dmp_op[DIM-W+i] *= gamma_eta;
      
  }
}




void create_CAP(int W, int C, int LE, r_type eta, r_type Emin, r_type dmp_op[]){
  r_type c=2.66206, gamma_eta=exp(-asinh(-eta));

  
  int SUBDIM = LE*W,
    DIM = (2*C+LE)*W;

 int ctLe  = C*W;

#pragma omp parallel for
  for(int i=0;i<SUBDIM;i++)
    dmp_op[i+C*W] = gamma_eta;

#pragma omp parallel for  
  for(int i=0;i<C*W;i++){
    dmp_op[i] = 1.0;
    dmp_op[DIM-i] = 1.0;    
  }


#pragma omp parallel for  
  for(int i=0;i<ctLe;i++){
    r_type gamma;
    if(C!=0){
      r_type y     = (r_type)(i/W)/C; 
      r_type func  = (-4.0*Emin/(c*c))*(1/pow(1-y,2)+1/pow(1+y,2)-2)+eta;
      gamma = asinh(-func);
      gamma = exp(-gamma);
    }
    else
      gamma = gamma_eta;

    dmp_op[ctLe-i-1]       *= gamma; 
    dmp_op[DIM-ctLe+i]     *= gamma;
      
  }
}



