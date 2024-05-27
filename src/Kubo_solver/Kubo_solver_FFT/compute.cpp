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
#include "../time_station_2.hpp"




void Kubo_solver_FFT::compute(){

  time_station_2 solver_station;
  solver_station.start();
  

  
  //----------------Initializing the Device---------------//
  //------------------------------------------------------//
  time_station_2 hamiltonian_setup_time;
  hamiltonian_setup_time.start();
  

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
      R = parameters_.R_,
      D = parameters_.dis_real_;

  
  E_min /= a;
  eta   /= a;



  
  time_station_2 allocation_time;
  allocation_time.start();

  
  
  allocate_memory();
  reset_Chebyshev_buffers();

  cap_->create_CAP(W, C, LE,  dmp_op_);
  device_.damp(dmp_op_);

  Kubo_solver_FFT_postProcess postProcess( (*this) );

  
  allocation_time.stop("\n \nAllocation time:            ");


  



  
  for(int d = 1; d <= D; d++){


    time_station_2 total_csrmv_time;
    time_station_2 total_FFTs_time;
	
    device_.Anderson_disorder(dis_vec_);
    device_.update_dis(dis_vec_, dmp_op_);




    
    for(int r = 1; r <= R; r++){


      time_station_2 randVec_time;
      randVec_time.start();
      std::cout<<std::endl<< std::to_string( d * r)+"/"+std::to_string(D*R)+"-Vector/disorder realization;"<<std::endl;

      vec_base_->generate_vec_im( rand_vec_, r);       
      device_.rearrange_initial_vec( rand_vec_ ); //very hacky

      reset_data(r_data_);
      



      
      for(int s=0; s < num_parts; s++){

	
	std::cout<< "   -Section: "<<s+1<<"/"<<num_parts<<std::endl;


	

        time_station_2 csrmv_time_kets;
        csrmv_time_kets.start();
	
        polynomial_cycle_ket( kets_, s );

	csrmv_time_kets.stop("           Kets cycle time:            ");
        total_csrmv_time += csrmv_time_kets;



	
	time_station_2 csrmv_time_bras;
        csrmv_time_bras.start();
	
	polynomial_cycle( bras_, s );	

	csrmv_time_bras.stop("           Bras cycle time:            ");
        total_csrmv_time += csrmv_time_bras;

	

	 

	time_station_2 FFTs_time;
	FFTs_time.start();
	
	if( sym_formula_ == KUBO_GREENWOOD )
	  Greenwood_FFTs(bras_, kets_, r_data_, s);

        if( sym_formula_ == KUBO_BASTIN )
	  Bastin_FFTs(bras_, kets_, r_data_, s);	

	FFTs_time.stop("           FFT operations time:        ");
	total_FFTs_time += FFTs_time;
	
	
	
      }

      
      total_csrmv_time.stop( "\n       Total CSRMV time:           ");
      total_FFTs_time.stop("       Total FFTs time:            ");





      time_station_2 time_postProcess;


      postProcess(final_data_, r_data_, r);
      /*
      if( sym_formula_ == KUBO_GREENWOOD )
        Greenwood_postProcess( ( d - 1 ) * R + r );

      if( sym_formula_ == KUBO_BASTIN )
        Bastin_postProcess( ( d - 1 ) * R + r );
      */
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

