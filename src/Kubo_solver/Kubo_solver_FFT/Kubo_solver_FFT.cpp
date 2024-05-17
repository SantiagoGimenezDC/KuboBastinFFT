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
#include "Kubo_solver_FFT.hpp"
#include "../time_station.hpp"


Kubo_solver_FFT::Kubo_solver_FFT(solver_vars& parameters, Device& device) : parameters_(parameters), device_(device)
{

  if(parameters_.cap_choice_ == 0)
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else if(parameters_.cap_choice_==1)
    cap_      = new Effective_Contact(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);


  
  if(parameters_.base_choice_ == 0)
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 1 )
    vec_base_ = new Complex_Phase(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 2 )
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



Kubo_solver_FFT::~Kubo_solver_FFT(){

  int M      = parameters_.M_;
  
/*------------Delete everything--------------*/
  //Single Shot vectors
  for(int m=0;m<M;m++){
    delete []bras_[m];
    delete []kets_[m];
  }
  delete []bras_;
  delete []kets_;

  //Recursion Vectors
  delete []vec_;
  delete []p_vec_;
  delete []pp_vec_;
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


void Kubo_solver_FFT::allocate_memory(){

  int M        = parameters_.M_,
      R        = parameters_.R_,
      D        = parameters_.dis_real_,
      DIM      = device_.parameters().DIM_,
      SUBDIM   = device_.parameters().SUBDIM_,
      num_p    = parameters_.num_p_,
      SEC_SIZE = SUBDIM / parameters_.num_parts_;

  parameters_.SECTION_SIZE_=SEC_SIZE;
  
/*------------Big memory allocation--------------*/
  //Single Shot vectors
  bras_ = new type* [ M ];
  kets_ = new type* [ M ];

  for(int m = 0; m < M; m++){
    bras_[m] = new type [ SEC_SIZE ];
    kets_[m] = new type [ SEC_SIZE ];
  }

  
  //Recursion Vectors
  vec_      = new type [ DIM ];
  p_vec_    = new type [ DIM ];
  pp_vec_   = new type [ DIM ];
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

  for(int k = 0; k < 2 * D * R; k++){
    conv_R_[k] = 0.0;
  }



  r_type buffer_mem    = r_type( 2 * r_type(M) * r_type(SEC_SIZE) * sizeof(type) ) / r_type( 1E9 ),
         recursion_mem = r_type( ( 5 * DIM + 1 * SUBDIM ) * sizeof(type) )/ r_type( 1E9 ),
         FFT_mem       = 0.0,
         Ham_mem = 2 * device_.Hamiltonian_size() / r_type( 1E9 ), //the 2 is because of the vel operator
         Total = 0.0;

  
  if(sym_formula_ == KUBO_GREENWOOD)
    FFT_mem = r_type( ( 1 + omp_get_num_threads() * ( 8 + 1 ) ) * num_p * sizeof(type) ) / r_type( 1E9 );
  if(sym_formula_ == KUBO_BASTIN)
    FFT_mem = r_type( ( 1 + omp_get_num_threads() * ( 16 + 1 ) ) * num_p * sizeof(type) ) / r_type( 1E9 );

  Total = buffer_mem + Ham_mem + recursion_mem + FFT_mem;

  
  std::cout<<std::endl;
  std::cout<<"Expected memory cost breakdown:"<<std::endl;
  std::cout<<"   Chebyshev buffers:    "<< buffer_mem<<" GBs"<<std::endl;  
  std::cout<<"   Operators size:     "<< Ham_mem<<" GBs"<<std::endl;  
  std::cout<<"   Recursion vectors:    "<<  recursion_mem <<" GBs"<<std::endl;
  std::cout<<"   FFT auxiliary lines:  "<<  FFT_mem <<" GBs"<<std::endl<<std::endl;   
  std::cout<<"TOTAL:  "<<  Total<<" GBs"<<std::endl<<std::endl;

}



void Kubo_solver_FFT::reset_Chebyshev_buffers(){
  int SEC_SIZE  = parameters_.SECTION_SIZE_,
      M         = parameters_.M_;

  for(int m = 0; m < M; m++){
#pragma omp parallel for	
    for(int l = 0; l < SEC_SIZE; l++){
      bras_[m][l] = 0.0;
      kets_[m][l] = 0.0;
    }
  }
}



void Kubo_solver_FFT::reset_recursion_vectors(){
  int DIM    = device_.parameters().DIM_;
  
#pragma omp parallel for	
   for(int k=0; k < DIM; k++){
     vec_     [k] = 0.0;
     pp_vec_  [k] = rand_vec_[k];
     p_vec_   [k] = 0.0;
   }
}


void Kubo_solver_FFT::reset_r_data(){
  int num_p    = parameters_.num_p_;
      
  for(int k=0; k< 2 * num_p; k++ )
    r_data_[k] = 0;
}







//LIST of former FFT operations

	//StandardProcess_Greenwood(bras,kets, E_points, r_data);


	//Bastin_FFTs__reVec_noEta_2(bras,kets, E_points, integrand);    	 
	//Bastin_FFTs__imVec_noEta_2(bras,kets, E_points, integrand);

	 

	//Greenwood_FFTs__reVec_noEta(bras,kets, E_points, r_data);         
	 

	/*
        These next 3 are meant to correct the pre factor. As it turns out, for small values of eta<0.1,
	those corrections are almost invisible and not worthwhile the huge increase in computational
	cost. The correction is meaningfull at the -1 and 1 edges, however, it is a lot more 
        reasonable to adjuste edge_ variable to deal with those 
	 
	//Bastin_FFTs__imVec_eta(bras,kets, E_points, integrand);
        //Greenwood_FFTs__reVec_eta(bras,kets, E_points, r_data);
	//Greenwood_FFTs__imVec_eta(bras,kets, E_points, r_data); 
        */
