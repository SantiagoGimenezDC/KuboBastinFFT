#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<eigen3/Eigen/Core>

#include "Kubo_solver_traditional.hpp"


void Kubo_solver_traditional::Bastin_postProcess (){
const std::complex<r_type> im(0,1);
 int M = parameters_.M_,
   num_p=parameters_.num_p_;
 
  Eigen::Matrix<type, -1,-1, Eigen::RowMajor> dgreenR(num_p,M),  local_integrand(1,num_p);
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor>  gamma_step(M,M), greenR(M,num_p), tmpMU(M,M), local_mu(M,M);

  r_type integrand[num_p];


  tmpMU = mu_.adjoint();  
  local_mu = ( mu_ + tmpMU ) / 2.0;




   greenR  = fill_green(E_points_, M, M);
   dgreenR = fill_dgreen(E_points_, M, M);

  
  gamma_step = local_mu * greenR;

  for(int e=0;e<num_p;e++)
    local_integrand(0,e) = 2.0 * 2.0 * M_PI * dgreenR.row(e) * gamma_step.col(e);
  
  

  for(int e=0;e<num_p;e++)
    integrand[e] = local_integrand(e).imag();


  
  r_type partial_result[num_p];
  
  integration(E_points_, integrand, partial_result);




  
  std::string run_dir  = parameters_.run_dir_,
              filename = parameters_.filename_;

  int SUBDIM=device_.parameters().SUBDIM_;
  r_type a = parameters_.a_,
         b = parameters_.b_,
         sysSubLength = device_.sysSubLength();

  r_type omega = SUBDIM/( a * a * sysSubLength * sysSubLength );//Dimensional and normalizing constant



  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e = 0; e < num_p; e++)  
    dataP<< a * E_points_[e] - b<<"  "<< omega * partial_result [e] <<std::endl;

  dataP.close();


  
  plot_data(run_dir, filename);

  
}


void Kubo_solver_traditional::Greenwood_postProcess(){
const std::complex<r_type> im(0,1);
 int M = parameters_.M_,
   num_p=parameters_.num_p_;

  Eigen::Matrix<type, -1,-1, Eigen::RowMajor> greenR2(M,num_p),  partial_result(1,num_p);
  Eigen::Matrix<type, -1,-1, Eigen::ColMajor>  gamma_step(M,num_p), greenR(num_p,M), tmpMU(M,M), local_mu(M,M);

  
  
  greenR  = fill_green(E_points_, M, num_p);

  tmpMU = mu_.adjoint();  
  local_mu = ( mu_ + tmpMU ) / 2.0;

  gamma_step = local_mu*greenR;

  //num_p-e-1??? WATT??
  for(int e=0;e<num_p;e++)
    partial_result(0,e) = - 2.0 * M_PI * M_PI * greenR.col(e).dot(gamma_step.col(e));
  

    std::string run_dir  = parameters_.run_dir_,
              filename = parameters_.filename_;


  int SUBDIM = device_.parameters().SUBDIM_;
    
  r_type a = parameters_.a_,
         b = parameters_.b_,
         sysSubLength = device_.sysSubLength();

  r_type omega = SUBDIM/( a * a * sysSubLength * sysSubLength );//Dimensional and normalizing constant


  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e = 0; e < num_p; e++)  
    dataP<< a * E_points_[e] - b<<"  "<< omega * partial_result(0,e).real() <<std::endl;

  dataP.close();


  
  plot_data(run_dir, filename);

  
}




type Kubo_solver_traditional::green(int n, type energy){
  const type i(0.0,1.0); 
  type sq = sqrt(1.0 - energy*energy);
  return -2.0/sq*i*exp(-type(n)*acos(energy)*i);
}

type Kubo_solver_traditional::dgreen(int n, type energy){
  const type i(0.0,1.0); 
  
  type den = 1.0 - energy*energy;
  type  sq = sqrt(den);
  return -2.0/den*i*exp(-r_type(n)*acos(energy)*i)*(type(n)*i + energy/sq);
}


Eigen::Matrix<type, -1,-1, Eigen::ColMajor> Kubo_solver_traditional::fill_green(r_type E_points[], int M, int E){
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


Eigen::Matrix<type, -1,-1, Eigen::RowMajor> Kubo_solver_traditional::fill_dgreen(r_type E_points[], int M, int E){

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



void Kubo_solver_traditional::integration(r_type E_points[], r_type integrand[], r_type data[]){

  int M     = parameters_.M_,
      num_p = parameters_.num_p_;
  
  r_type edge = parameters_.edge_;
  
#pragma omp parallel for 
  for(int k=0; k<num_p - int( M * edge / 4.0 ); k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                       //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                      //the number of points to be ignored in the future;
    for(int j=k; j<num_p-int(M*edge/4.0); j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      r_type ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ;
    }
  }
}


void Kubo_solver_traditional::partial_integration(r_type E_points[], r_type integrand[], r_type data[]){

  int M     = parameters_.M_,
      end   = M * 0.55,
      start = M * 0.45;
  
#pragma omp parallel for 
  for(int k=start; k<end; k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                       //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                      //the number of points to be ignored in the future;
    for(int j=k; j<end; j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      r_type ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ;
    }
  }
}
     

void Kubo_solver_traditional::rearrange_crescent_order( r_type* rearranged){//The point choice of the FFT has unconvenient ordering for file saving and integrating; This fixes that.
  int nump = parameters_.num_p_;
  r_type original[nump];

  for( int k=0; k < nump; k++ )
    original[k] = rearranged[k];

  
  for( int k=0; k < nump / 2; k++ ){
    rearranged[ 2 * k ]   = original[ k ];
    rearranged[ 2 * k + 1 ] = original[ nump - k - 1 ]; 
  }
  
  for( int k=0; k < nump / 2; k++ ){
    r_type tmp = rearranged[ k ]; 
    rearranged[ k ]   = rearranged[ nump-k-1 ];
    rearranged[ nump-k-1 ] = tmp;    
  }  
}




void Kubo_solver_traditional::plot_data(std::string run_dir, std::string filename){
        //VIEW commands
  
     std::string exestring=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"

         "set output '"+run_dir+filename+".png'                \n"

         "set xlabel 'E[eV]'                                               \n"
         "set ylabel  'G [2e^2/h]'                                           \n"
         
        "plot '"+run_dir+"currentResult_"+filename+"' using 1:2 w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
      char exeChar[exestring.size() + 1];
      strcpy(exeChar, exestring.c_str());    
      if(system(exeChar)){};


}

