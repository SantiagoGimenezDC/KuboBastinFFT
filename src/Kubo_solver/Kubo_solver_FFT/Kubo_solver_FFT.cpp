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


void Kubo_solver_FFT::initialize_device(){

  device_.build_Hamiltonian();
  device_.setup_velOp();
  
  if(parameters_.a_ == -1.0){
    r_type Emin, Emax;
    device_.minMax_EigenValues(300, Emax,Emin);

    
    parameters_.a_ =  ( Emax - Emin ) / ( 2.0 - parameters_.edge_ );
    parameters_.b_ = -( Emax + Emin ) / 2.0;

  }
  
  device_.adimensionalize( parameters_.a_, parameters_.b_ );
}



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
  else if(parameters_.base_choice_ == 3 )
    vec_base_ = new FullTrace(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 4 )
    vec_base_ = new projected_FullTrace(device_.parameters(), parameters_.seed_, device_.unit_cell_size());




  
  if(parameters_.kernel_choice_ == 0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();
  else
    kernel_   = new None();

  
  parameters_.SECTION_SIZE_ = device_.parameters().SUBDIM_ / parameters_.num_parts_ + device_.parameters().SUBDIM_ % parameters_.num_parts_;
  
  sym_formula_ = parameters_.sim_equation_;
}



Kubo_solver_FFT::~Kubo_solver_FFT(){

  int M      = parameters_.M_;
  
/*------------Delete everything--------------*/

  //Recursion Vectors
  delete []rand_vec_;
  delete []tmp_;
  
  //Auxiliary - disorder and CAP vectors
  delete []dmp_op_;

  //Single Shot vectors
  for(int m=0;m<M;m++){
    delete []bras_[m];
    delete []kets_[m];
  }
  delete []bras_;
  delete []kets_;

/*-----------------------------------------------*/

  delete kernel_;
  delete cap_;
  delete vec_base_;
}


void Kubo_solver_FFT::allocate_memory(){

  int M        = parameters_.M_,
      DIM      = device_.parameters().DIM_,
      SUBDIM   = device_.parameters().SUBDIM_,
      num_p    = parameters_.num_p_,
      SEC_SIZE = SUBDIM / parameters_.num_parts_;

  parameters_.SECTION_SIZE_ = SEC_SIZE;


  r_type buffer_mem    = r_type( 2 * r_type(M) * r_type(SEC_SIZE) * sizeof(type) ) / r_type( 1E9 ),
         recursion_mem = r_type( ( 5 * DIM + 1 * SUBDIM ) * sizeof(type) )/ r_type( 1E9 ),
         FFT_mem       = 0.0,
         Ham_mem = 2 * device_.Hamiltonian_size() / r_type( 1E9 ), //the 2 is because of the vel operator
         Total = 0.0;


  if(sym_formula_ == KUBO_GREENWOOD)
    FFT_mem = r_type( ( 1 + omp_get_num_threads() * ( 8 + 1 ) ) * num_p * sizeof(type) ) / r_type( 1E9 );
  if(sym_formula_ == KUBO_BASTIN || sym_formula_ == KUBO_SEA )
    FFT_mem = r_type( ( 1 + omp_get_num_threads() * ( 16 + 1 ) ) * num_p * sizeof(type) ) / r_type( 1E9 );

  Total = buffer_mem + Ham_mem + recursion_mem + FFT_mem;

  
  std::cout<<std::endl;
  std::cout<<"Expected memory cost breakdown:"<<std::endl;
  std::cout<<"   Chebyshev buffers:    "<< buffer_mem<<" GBs"<<std::endl;  
  std::cout<<"   Operators size:       "<< Ham_mem<<" GBs"<<std::endl;  
  std::cout<<"   Recursion vectors:    "<<  recursion_mem <<" GBs"<<std::endl;
  std::cout<<"   FFT auxiliary lines:  "<<  FFT_mem <<" GBs"<<std::endl<<std::endl;   
  std::cout<<"TOTAL:  "<<  Total<<" GBs"<<std::endl<<std::endl;

  
  
/*------------Big memory allocation--------------*/
  //Single Shot vectors
  
  bras_ = new type* [ M ];
  kets_ = new type* [ M ];

  for(int m = 0; m < M; m++){
    bras_[m] = new type [ SEC_SIZE ];
    kets_[m] = new type [ SEC_SIZE ];
  }
  
  
  //Recursion Vectors
  rand_vec_ = new type [ DIM ];
  tmp_      = new type [ DIM ];
  
  //Disorder and CAP vectors
  dmp_op_  = new r_type [ DIM ];
/*-----------------------------------------------*/



  
/*---------------Dataset vectors----------------*/

  r_data_.resize( 2 * num_p );  
  final_data_.resize( 2 * num_p );
  
/*-----------------------------------------------*/  

  
#pragma omp parallel for	
  for(int k=0; k < DIM; k++){
    dmp_op_   [k] = 1.0;
    rand_vec_ [k] = 0.0;
  }
    

  reset_data( r_data_ );
  reset_data( final_data_ );  


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






void Kubo_solver_FFT::update_data(std::vector<type>& final_data, const std::vector<type>& new_r_data, int  r){
  int nump    = parameters_.num_p_, end = 0;

  if( sym_formula_ == KUBO_GREENWOOD )
    end = nump;
  if( sym_formula_ == KUBO_BASTIN || sym_formula_ == KUBO_SEA )
    end = 2 * nump;

  
  for(int i = 0; i < end; i++)
      final_data[i] = ( final_data[i] * type( r - 1 ) + new_r_data[i] ) / type(r);
};




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
