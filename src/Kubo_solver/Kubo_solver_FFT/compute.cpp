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




void Kubo_solver_FFT::compute(){

  time_station solver_station;

  
  //----------------Initializing the Device---------------//
  time_station hamiltonian_setup_time;

  device_.build_Hamiltonian();
  device_.setup_velOp();
  
  if(parameters_.a_ == 1.0){
    r_type Emin, Emax;
    device_.minMax_EigenValues(300, Emax,Emin);


    parameters_.a_ =  ( Emax - Emin ) / ( 2.0 - parameters_.edge_ );
    parameters_.b_ = -( Emax + Emin ) / 2.0;
  }
  
  device_.adimensionalize( parameters_.a_, parameters_.b_ );

  hamiltonian_setup_time.stop("    Time to setup the Hamiltonian:            ");
  std::cout<<std::endl;
  //------------------------------------------------------//
  

  
  int W      = device_.parameters().W_,
      C      = device_.parameters().C_,
      LE     = device_.parameters().LE_;

  r_type a       = parameters_.a_,
         E_min   = parameters_.E_min_,
    //   E_start = parameters_.E_start_,
    //   E_end   = parameters_.E_end_,
         eta     = parameters_.eta_;
  
  int num_parts = parameters_.num_parts_,
      num_p = parameters_.num_p_,
      R = parameters_.R_,
      D = parameters_.dis_real_;

  std::string run_dir  = parameters_.run_dir_,
              filename = parameters_.filename_;

  E_min /= a;
  eta   /= a;



  
  time_station allocation_time;

  allocate_memory();
  reset_Chebyshev_buffers();

  cap_->create_CAP(W, C, LE,  dmp_op_);
  device_.damp(dmp_op_);

  allocation_time.stop("\n \nAllocation time:            ");





  
  for(int d = 1; d <= D; d++){


    
    int total_time_csrmv = 0,
        total_time_FFTs  = 0;

    device_.Anderson_disorder(dis_vec_);
    device_.update_dis(dis_vec_, dmp_op_);




    
    for(int r = 1; r <= R; r++){


      time_station randVec_time;
      std::cout<<std::endl<< std::to_string( d * r)+"/"+std::to_string(D*R)+"-Vector/disorder realization;"<<std::endl;


      
      vec_base_->generate_vec_im( rand_vec_, r);       
      device_.rearrange_initial_vec( rand_vec_ ); //very hacky
    
      for(int k=0; k< 2 * num_p; k++ )
        r_data_[k] = 0;

      

      
      
      for(int s=0; s < num_parts; s++){


	
	std::cout<< "   -Section: "<<s+1<<"/"<<num_parts<<std::endl;


	
	time_station csrmv_time_kets;

        polynomial_cycle_ket( kets_, s );

	csrmv_time_kets.stop_add( &total_time_csrmv, "           Kets cycle time:            ");




	time_station csrmv_time_bras;

	polynomial_cycle( bras_, s );	

	csrmv_time_bras.stop_add( &total_time_csrmv,  "           Bras cycle time:            ");


	

	 

	time_station FFTs_time;

	if( sym_formula_ == KUBO_GREENWOOD )
	  Greenwood_FFTs__imVec(bras_, kets_, r_data_, s);

        if( sym_formula_ == KUBO_BASTIN )
	  Bastin_FFTs(bras_, kets_, r_data_, s);	

	FFTs_time.stop_add( &total_time_FFTs, "           FFT operations time:        ");

	
	
	
      }

      
      time_station total_CSRMV(total_time_csrmv, "\n       Total CSRMV time:           ");
      time_station total_FFTs(total_time_FFTs, "       Total FFTs time:            ");





      time_station time_postProcess;

      if( sym_formula_ == KUBO_GREENWOOD )
        Greenwood_postProcess( ( d - 1 ) * R + r );

      if( sym_formula_ == KUBO_BASTIN )
        Bastin_postProcess( ( d - 1 ) * R + r );

      time_postProcess.stop( "       Post-processing time:       ");

      

      
      randVec_time.stop("       Total RandVec time:         ");
      std::cout<<std::endl;
    }
  }


  
  solver_station.stop("Total case execution time:              ");
}




void Kubo_solver_FFT::polynomial_cycle(type** polys, int s){
  
  int M = parameters_.M_,
      num_parts = parameters_.num_parts_;
  
  
  reset_recursion_vectors();

//=================================KPM Step 0======================================//


  device_.traceover(polys[0], pp_vec_, s, num_parts);
  

  
//=================================KPM Step 1======================================//   
    
  
  device_.H_ket ( p_vec_, pp_vec_, dmp_op_, dis_vec_);

  device_.traceover(polys[1], p_vec_, s, num_parts);
    
    

//=================================KPM Steps 2 and on===============================//
    
  for( int m = 2; m < M; m++ ){
    device_.update_cheb( vec_, p_vec_, pp_vec_, dmp_op_, dis_vec_);
    device_.traceover(polys[m], vec_, s, num_parts);
  }
}




void Kubo_solver_FFT::polynomial_cycle_ket(type** polys,  int s){

  int M   = parameters_.M_,
      num_parts = parameters_.num_parts_;

  reset_recursion_vectors();
  device_.vel_op( pp_vec_, rand_vec_ );


//=================================KPM Step 0======================================//
  
  device_.vel_op( tmp_, pp_vec_ );
  device_.traceover(polys[0], tmp_, s, num_parts);  
  

  
//=================================KPM Step 1======================================//       
    
  device_.H_ket ( p_vec_, pp_vec_, dmp_op_, dis_vec_);
  device_.vel_op( tmp_, p_vec_ );
  
  device_.traceover(polys[1], tmp_, s, num_parts);  
  
    

//=================================KPM Steps 2 and on===============================//
    
  for( int m = 2; m < M; m++ ){
    device_.update_cheb( vec_, p_vec_, pp_vec_, dmp_op_, dis_vec_);
    device_.vel_op( tmp_, vec_ );
    device_.traceover(polys[m], tmp_, s, num_parts);
  }
}

