#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<eigen-3.4.0/Eigen/Core>

#include "Kubo_solver.hpp"




void Kubo_solver::StandardProcess_Bastin (std::complex<r_type> **bras, std::complex<r_type> **kets, r_type E_points[], r_type integrand[]){
const std::complex<r_type> im(0,1);
 int M = parameters_.M_,
   num_p=parameters_.num_p_,
   SUBDIM = device_.parameters().SUBDIM_;
 
  Eigen::Matrix<type, -1,-1, Eigen::RowMajor> dgreenR(num_p,M),  local_integrand(1,num_p);
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor>  gamma_step(M,M), greenR(M,num_p), tmpMU(M,M), mu(M,M);



  mu.setZero();
#pragma omp parallel for
  for(int j=0;j<M;j++){
    
    Eigen::Matrix<type, -1,-1, Eigen::ColMajor>  local_mu(M,M);

    local_mu.setZero();
    
    for(int i=0; i<M;i++)
      for(int k=0; k<SUBDIM;k++)
        local_mu(i,j) +=  conj(bras[i][k])  * kets[j][k];

#pragma omp critical
    {
     for(int i=0;i<M;i++)
       for(int j=0;j<M;j++)
	 mu(i,j)+=local_mu(i,j);
    }
    }

  
    mu=(mu+tmpMU)/2.0;




   
   greenR  = fill_green(E_points, M, M);
   dgreenR = fill_dgreen(E_points, M, M);

  
  gamma_step = mu*greenR;

  for(int e=0;e<num_p;e++)
    local_integrand(0,e) = -2.0*2.0*M_PI*dgreenR.row(e)*gamma_step.col(e);
  
  

  for(int e=0;e<num_p;e++){
    //    if(sqrt(E_points[e]*E_points[e])<0.99 && sqrt(E_points[e]*E_points[e])>0.01)
    integrand[e]=local_integrand(e).imag();
    //else
    //integrand[e]=0.0;


    /*
 if(parameters_.eta_!=0)
   for(int j=0; j<M; j++ )
     integrand[j] *= sin(acos(E_points[j]));
    */

  }
}


void Kubo_solver::StandardProcess_Greenwood(std::complex<r_type> **bras, std::complex<r_type> **kets, r_type E_points[], r_type integrand[]){
const std::complex<r_type> im(0,1);
 int M = parameters_.M_,
   num_p=parameters_.num_p_,
   SUBDIM = device_.parameters().SUBDIM_;

  Eigen::Matrix<type, -1,-1, Eigen::RowMajor> greenR2(M,num_p),  local_integrand(1,num_p);
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor>  gamma_step(M,num_p), greenR(num_p,M), mu(M,M), tmpMU(M,M);


  mu.setZero();
#pragma omp parallel for
  for(int j=0;j<M;j++){
    
    Eigen::Matrix<type, -1,-1, Eigen::ColMajor>  local_mu(M,M);

    local_mu.setZero();
    
    for(int i=0; i<M;i++)
      for(int k=0; k<SUBDIM;k++)
        local_mu(i,j) +=  conj(bras[i][k])  * kets[j][k];

#pragma omp critical
    {
     for(int i=0;i<M;i++)
       for(int j=0;j<M;j++)
	 mu(i,j)+=local_mu(i,j);
    }
    }

    
  tmpMU= mu.adjoint();
  mu = ((mu+tmpMU)/2.0);
  
  greenR  = fill_green(E_points, M, num_p);


  gamma_step = mu*greenR;

  //num_p-e-1??? WATT??
  for(int e=0;e<num_p;e++)
    local_integrand(0,num_p-e-1) = 2.0*M_PI*M_PI*greenR.col(e).dot(gamma_step.col(e));
  
  
  
  for(int e=0;e<num_p;e++)
    integrand[e]=local_integrand(e).real();//*sin(acos(E_points[e]) );
   
}




type Kubo_solver::green(int n, type energy){
  const type i(0.0,1.0); 
  type sq = sqrt(1.0 - energy*energy);
  return -2.0/sq*i*exp(-type(n)*acos(energy)*i);
}

type Kubo_solver::dgreen(int n, type energy){
  const type i(0.0,1.0); 
  
  type den = 1.0 - energy*energy;
  type  sq = sqrt(den);
  return -2.0/den*i*exp(-r_type(n)*acos(energy)*i)*(type(n)*i + energy/sq);
}


Eigen::Matrix<type, -1,-1, Eigen::ColMajor> Kubo_solver::fill_green(r_type E_points[], int M, int E){
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor> greenR = Eigen::Matrix<type, -1,-1, Eigen::ColMajor>::Zero(M, E);

  r_type a = parameters_.a_,
         eta = parameters_.eta_/a;
  

  
  r_type factor;
  type complexEnergy;
  for(int e = 0; e < E; e++){
                eta = ( parameters_.eta_/a ) * sin(acos(E_points[e])); 
    complexEnergy = type(E_points[e], eta);
    for(int m = 0; m < M; m++){
      factor = -1.0/(1.0 + (m==0))/M_PI*kernel_->term(m,M);
      greenR(m, e) = green(m, complexEnergy).imag()*factor;
    }
  }
  return greenR;
}


Eigen::Matrix<type, -1,-1, Eigen::RowMajor> Kubo_solver::fill_dgreen(r_type E_points[], int M, int E){

  Eigen::Matrix<type, -1,-1, Eigen::RowMajor> dgreenR = Eigen::Matrix<type, -1,-1, Eigen::RowMajor>::Zero(M, E);

  r_type a = parameters_.a_,
         eta = parameters_.eta_/a; 
  r_type factor;
  type complexEnergy;

  
  for(int e = 0; e < E; e++){
    eta = ( parameters_.eta_/a ) * sin(acos(E_points[e])); 
    complexEnergy = type(E_points[e], eta);
    for(int m = 0; m < M; m++){
      factor = 1.0/(1.0 + (m==0))*kernel_->term(m,M);
      dgreenR(e, m) = dgreen(m, complexEnergy)*factor;
    }
  }
  return dgreenR;
}
