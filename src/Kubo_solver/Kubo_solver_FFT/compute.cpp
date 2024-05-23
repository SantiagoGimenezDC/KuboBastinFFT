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

  int M        = parameters_.M_,
      DIM   = device_.parameters().DIM_,
      SUBDIM   = device_.parameters().SUBDIM_,
      SEC_SIZE = SUBDIM / parameters_.num_parts_;

  parameters_.SECTION_SIZE_=SEC_SIZE;

  E_min /= a;
  eta   /= a;




  
  
  time_station_2 allocation_time;
  allocation_time.start();

  
  Eigen_states_buffer<type> bras(SEC_SIZE, M);
  Eigen_states_buffer<type> kets(SEC_SIZE, M);
  
  Chebyshev_states< Eigen_state<type> > Chebyshev_vectors(device_);

  
  allocate_memory();

  bras.reset();
  kets.reset();

  cap_->create_CAP(W, C, LE,  dmp_op_);
  device_.damp(dmp_op_);

  
  allocation_time.stop("\n \nAllocation time:            ");



  
  



  
  for(int d = 1; d <= D; d++){


    time_station_2 total_csrmv_time;
    time_station_2 total_FFTs_time;
	
    device_.Anderson_disorder( dis_vec_ );
    device_.update_dis( dis_vec_, dmp_op_);




    
    for(int r = 1; r <= R; r++){


      time_station_2 randVec_time;
      randVec_time.start();
      std::cout<<std::endl<< std::to_string( d * r)+"/"+std::to_string(D*R)+"-Vector/disorder realization;"<<std::endl;

      
      vec_base_->generate_vec_im( rand_vec_, r);       
      device_.rearrange_initial_vec( rand_vec_ ); //very hacky
      Eigen::Vector<type, Eigen::Dynamic> pass_data = Eigen::Map< Eigen::Vector<type, Eigen::Dynamic> > (rand_vec_, DIM);
      Eigen_state< type > rand_vec ( pass_data );
  
      
      reset_r_data();
      



      
      for(int s=0; s < num_parts; s++){

	
	std::cout<< "   -Section: "<<s+1<<"/"<<num_parts<<std::endl;


	

        time_station_2 csrmv_time_kets;
        csrmv_time_kets.start();

	Chebyshev_vectors.reset(rand_vec);
        polynomial_cycle( kets, Chebyshev_vectors, s, false );

	csrmv_time_kets.stop("           Kets cycle time:            ");
        total_csrmv_time += csrmv_time_kets;



	
	time_station_2 csrmv_time_bras;
        csrmv_time_bras.start();

	Chebyshev_vectors.reset(rand_vec);
	polynomial_cycle( bras, Chebyshev_vectors, s, true); 
	
	csrmv_time_bras.stop("           Bras cycle time:            ");
        total_csrmv_time += csrmv_time_bras;

	

	 

	time_station_2 FFTs_time;
	FFTs_time.start();
	
	if( sym_formula_ == KUBO_GREENWOOD )
	  Greenwood_FFTs__imVec(bras, kets, r_data_, s);

        if( sym_formula_ == KUBO_BASTIN )
	  Bastin_FFTs(bras, kets, r_data_, s);	

	FFTs_time.stop("           FFT operations time:        ");
	total_FFTs_time += FFTs_time;
	
	
	
      }

      
      total_csrmv_time.stop( "\n       Total CSRMV time:           ");
      total_FFTs_time.stop("       Total FFTs time:            ");





      time_station_2 time_postProcess;

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



void Kubo_solver_FFT::polynomial_cycle(Eigen_states_buffer<type> vector_buffer, Chebyshev_states<Eigen_state<type> > Chebyshev_vectors, int s, bool apply_vel){
  
  int M = parameters_.M_,
      num_parts = parameters_.num_parts_;

  type* tmp = new type[Chebyshev_vectors[0].state_data().size()];  

  
  if(apply_vel){
    device_.vel_op( tmp, Chebyshev_vectors[0].state_data().data() );
    device_.traceover(vector_buffer[0].state_data().data(), tmp, s, num_parts);
  }
  else
    device_.traceover(vector_buffer[0].state_data().data(), Chebyshev_vectors[0].state_data().data(), s, num_parts);
  



  
  for( int m = 1; m < M; m++ ){

    Chebyshev_vectors.update();

    if(apply_vel){
      device_.vel_op( tmp, Chebyshev_vectors[0].state_data().data() );
      device_.traceover(vector_buffer[m].state_data().data(), tmp, s, num_parts );
    }   
    else
      device_.traceover(vector_buffer[m].state_data().data(), Chebyshev_vectors[0].state_data().data(), s, num_parts);
  }
}

