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
#include "Kubo_solver_traditional.hpp"
#include "../time_station.hpp"


Kubo_solver_traditional::Kubo_solver_traditional(solver_vars& parameters, Device& device) : parameters_(parameters), device_(device)
{

  if(parameters_.cap_choice_ == 0)
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else if(parameters_.cap_choice_==1)
    cap_      = new Effective_Contact(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);


  
  if(parameters_.base_choice_ == 0)
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_==1)
    vec_base_ = new Complex_Phase(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_==2)
    vec_base_ = new Complex_Phase_real(device_.parameters(), parameters_.seed_);
  else
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);

  
  if(parameters_.kernel_choice_ == 0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();
  else
    kernel_   = new None();

  
  parameters_.SECTION_SIZE_ = device_.parameters().SUBDIM_ / parameters_.num_parts_ + device_.parameters().SUBDIM_ % parameters_.num_parts_;
  
  sym_formula_ = KUBO_GREENWOOD;
}


Kubo_solver_traditional::~Kubo_solver_traditional(){

/*------------Delete everything--------------*/

  //Recursion Vectors
  delete []vec_;
  delete []p_vec_;
  delete []pp_vec_;

  delete []vec_2_;
  delete []p_vec_2_;
  delete []pp_vec_2_;

  delete []rand_vec_;
  delete []tmp_;
  
  //Auxiliary - disorder and CAP vectors
  delete []dmp_op_;
  delete []dis_vec_;
/*-----------------------------------------------*/

  delete kernel_;
  delete cap_;
  delete vec_base_;
}


void Kubo_solver_traditional::allocate_memory(){

  int M        = parameters_.M_,
      R        = parameters_.R_,
      D        = parameters_.dis_real_,
      DIM      = device_.parameters().DIM_,
      SUBDIM   = device_.parameters().SUBDIM_,
      num_p    = parameters_.num_p_,
      num_parts = parameters_.num_parts_,
      SEC_M = M / num_parts,
      buffer_size = SEC_M + M % num_parts;


  
/*------------Big memory allocation--------------*/
  //Moments matrix:
  mu_.resize(M,M);
  mu_r_.resize(M,M);

  //Single Shot vectors
  bras_.resize(SUBDIM, buffer_size);
  kets_.resize(SUBDIM, buffer_size);
  
  //Recursion Vectors; 4 of these are actually not needed;
  vec_      = new type [ DIM ];
  p_vec_    = new type [ DIM ];
  pp_vec_   = new type [ DIM ];

  vec_2_      = new type [ DIM ];
  p_vec_2_    = new type [ DIM ];
  pp_vec_2_   = new type [ DIM ];
  
  rand_vec_ = new type [ DIM ];
  tmp_      = new type [ DIM ];
  
  //Disorder and CAP vectors
  dmp_op_  = new r_type [ DIM ],
  dis_vec_ = new r_type [ SUBDIM ];
/*-----------------------------------------------*/



  
/*---------------Dataset vectors----------------*/
  E_points_ = new r_type [ num_p ];
  r_data_     = new type [ 2 * num_p ];  
  final_data_ = new type [ 2 * num_p ];
  
  conv_R_   = new r_type [ 2 * D * R ];
/*-----------------------------------------------*/  

  
#pragma omp parallel for	
  for(int k=0; k < DIM; k++){
    dmp_op_   [k] = 1.0;
    rand_vec_ [k] = 0.0;
  }
    
#pragma omp parallel for	
  for(int k=0; k < SUBDIM; k++)  
    dis_vec_  [k] = 0.0;


  
  
  for(int k = 0; k<num_p; k++){
    E_points_[k]   = cos(  M_PI * (  2.0 * r_type(k) + 0.5 ) / r_type (num_p) ); 

    r_data_[k]     = 0.0;
    final_data_[k] = 0.0;
    
    r_data_[k + num_p]     = 0.0;
    final_data_[k + num_p] = 0.0;
  }

  rearrange_crescent_order(E_points_);

  for(int k = 0; k < 2 * D * R; k++){
    conv_R_[k] = 0.0;
  }



  r_type buffer_mem    = r_type( 2 * SEC_M * SUBDIM * sizeof(type) ) / r_type( 1000000000 ),
         recursion_mem = r_type( ( 7 * DIM + 1 * SUBDIM ) * sizeof(type) )/ r_type( 1000000000 ),
         mu_mem        = 2 * M * M * sizeof(type) / r_type( 1000000000 ),
         Ham_mem       = device_.Hamiltonian_size()/ r_type( 1000000000 ),
         Total         = 0.0;

 
  Total = buffer_mem + Ham_mem + recursion_mem + mu_mem;

  
  std::cout<<std::endl;
  std::cout<<"Expected memory cost breakdown:"<<std::endl;
  std::cout<<"   Chebyshev buffers:    "<< buffer_mem<<" GBs"<<std::endl;  
  std::cout<<"   Hamiltonian size:     "<< Ham_mem<<" GBs"<<std::endl;  
  std::cout<<"   Recursion vectors:    "<<  recursion_mem <<" GBs"<<std::endl;
  std::cout<<"   Moments Matrix:       "<<  mu_mem <<" GBs"<<std::endl;
  std::cout<<"TOTAL:  "<<  Total<<" GBs"<<std::endl<<std::endl;

}



void Kubo_solver_traditional::reset_Chebyshev_buffers(){
  
  bras_.setZero();
  kets_.setZero();

  mu_r_.setZero();
  mu_.setZero();
}



void Kubo_solver_traditional::reset_ket_recursion_vectors(){
  int DIM    = device_.parameters().DIM_;
  
#pragma omp parallel for	
   for(int k=0; k < DIM; k++){
     vec_     [k] = 0.0;
     pp_vec_  [k] = rand_vec_[k];
     p_vec_   [k] = 0.0;
   }
}


void Kubo_solver_traditional::reset_bra_recursion_vectors(){
  int DIM    = device_.parameters().DIM_;
  
#pragma omp parallel for	
   for(int k=0; k < DIM; k++){
     vec_2_     [k] = 0.0;
     pp_vec_2_  [k] = rand_vec_[k];
     p_vec_2_   [k] = 0.0;
   }
}





