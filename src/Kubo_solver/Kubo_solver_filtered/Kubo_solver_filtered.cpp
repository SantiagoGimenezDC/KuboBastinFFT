#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<chrono>
#include<ctime>
#include<omp.h>
#include<iomanip>
#include<cstdlib>
#include<thread>
#include<complex>
#include<cstring>

#include<fftw3.h>



#include "../../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"

#include "../time_station.hpp"



Kubo_solver_filtered::Kubo_solver_filtered(solver_vars& parameters, Device& device, KB_filter& filter) : parameters_(parameters), device_(device), filter_(filter)
{

  if(parameters_.cap_choice_==0)
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else if(parameters_.cap_choice_==1)
    cap_      = new Effective_Contact(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);


  
  if(parameters_.base_choice_ == 0 )
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 1 )
    vec_base_ = new Complex_Phase(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 2 )
    vec_base_ = new Complex_Phase_real(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 3 )
    vec_base_ = new FullTrace(device_.parameters(), parameters_.seed_);

  
  if(parameters_.kernel_choice_==0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();
  else
    kernel_   = new None();

  parameters_.SECTION_SIZE_ = device_.parameters().SUBDIM_ / parameters_.num_parts_ + device_.parameters().SUBDIM_ % parameters_.num_parts_;

  sym_formula_ = parameters_.sim_equation_;//KUBO_SEA;
  
}



void Kubo_solver_filtered::reset_buffer(type** polys){
    
  int M_dec = filter_.M_dec(),
    SEC_SIZE    =   parameters_.SECTION_SIZE_;

#pragma omp parallel for
  for(int m = 0; m < M_dec; m++)
    for(int i = 0; i < SEC_SIZE; i++)
      polys[m][i] = 0.0;
};

void Kubo_solver_filtered::compute_E_points( r_type* E_points ){
      
  int k_dis = filter_.parameters().k_dis_,
      M_ext = filter_.parameters().M_ext_,
      nump = filter_.parameters().nump_;	


  if( nump % 2 == 1 ){
    for(int k = 0; k < ( nump - 1 ) / 2 ; k++){
      E_points[ k ]                = cos( 2.0 * M_PI * ( k - k_dis + 0.25 ) / double(M_ext) );             
      E_points[ ( nump - 1 ) / 2 + k + 1 ] = cos( 2.0 * M_PI * ( ( nump - 1 ) / 2 - ( k - k_dis + 0.25 ) )  / double(M_ext) );
    }
    E_points[ ( nump - 1 ) / 2 ]           = cos( 2 * M_PI * ( ( nump - 1 ) / 2 - ( - 1 - k_dis + 0.25 ) )  / double(M_ext) );             
  }
  else
    for(int k = 0; k < nump/2 ; k++){
      E_points[ k ]            = cos( 2 * M_PI * ( k - k_dis + 0.25 ) / M_ext );             
      E_points[ nump / 2 + k ] = cos( 2 * M_PI * ( nump / 2 - ( k - k_dis + 0.25 ) )  / double(M_ext) );
    }
  
};


void Kubo_solver_filtered::batch_vel_op(std::complex<r_type>** buffer, int M , int SEC_SIZE){

  for(int m=0; m<M; m++ ){
    // device_.vel_op( tmp_velOp, new_vec );
  }

};












