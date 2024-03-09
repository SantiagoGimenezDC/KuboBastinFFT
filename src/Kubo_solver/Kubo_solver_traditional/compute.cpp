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




void Kubo_solver_traditional::compute(){

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
  
  int M = parameters_.M_,
      num_parts = parameters_.num_parts_,
      SEC_M = M / num_parts,
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


    
    int tmp_time_csrmv = 0,
        total_time_csrmv = 0,
        total_time_ZGEMM  = 0;

    device_.Anderson_disorder(dis_vec_);
    device_.update_dis(dis_vec_, dmp_op_);




    
    for(int r = 1; r <= R; r++){


      time_station randVec_time;
      std::cout<<std::endl<< std::to_string( d * r )+"/"+std::to_string(D*R)+"-Vector/disorder realization;"<<std::endl;


      
      vec_base_->generate_vec_im( rand_vec_, r);       
      device_.rearrange_initial_vec( rand_vec_ ); //very hacky

      
      for(int k=0; k< 2 * num_p; k++ )
        r_data_[k] = 0;


      reset_ket_recursion_vectors();
      device_.vel_op( pp_vec_, rand_vec_ );
  



      
      for(int s1 = 0; s1 < num_parts; s1++){


	std::cout<< "   -Section: "<<s1+1<<"/"<<num_parts<<std::endl;
	
	int s1_length = (s1 == num_parts-1 ? SEC_M : SEC_M + M % num_parts ) ;


	
	time_station csrmv_time_kets;

	polynomial_cycle_ket(s1 * SEC_M, s1 * SEC_M + s1_length);
	  
	csrmv_time_kets.stop_add( &tmp_time_csrmv );
        total_time_csrmv += tmp_time_csrmv;
	
	

	
	reset_bra_recursion_vectors();

	for(int s2 = 0; s2 < num_parts; s2++){
	    
	  int s2_length = (s2 == num_parts - 1 ? SEC_M : SEC_M + M % num_parts ) ;


	  
	  
	  time_station csrmv_time_bras;

	  polynomial_cycle_bra(s2 * SEC_M, s2 * SEC_M + s2_length);

	  csrmv_time_bras.stop_add_add(tmp_time_csrmv,  &total_time_csrmv,  "           Section poly cycle time:      ");


	  
	  
	  time_station ZGEMM_time;
		
	  update_cheb_moments(s1,s2, s1_length, s2_length);    

	  ZGEMM_time.stop_add( &total_time_ZGEMM, "           ZGEMM operations time:        ");

	}
      }
      
      






      

      
      randVec_time.stop("       Total RandVec time:         ");
      std::cout<<std::endl;
    }
  }


  
  time_station time_postProcess;

  if( sym_formula_ == KUBO_GREENWOOD )
    Greenwood_postProcess();

  if( sym_formula_ == KUBO_BASTIN )
    Bastin_postProcess();

  time_postProcess.stop( "       Post-processing time:       ");


  
  
  solver_station.stop("Total case execution time:              ");
}





void Kubo_solver_traditional::update_cheb_moments(int s1,int s2, int s1_length, int s2_length){
  int SUBDIM = device_.parameters().SUBDIM_,
      M = parameters_.M_,
      num_parts = parameters_.num_parts_,
      SEC_M = M / num_parts,
      size = SUBDIM;

  int Nthrds_backup = Eigen::nbThreads();

  Eigen::setNbThreads(1);
  //  mu_ = bras_.adjoint() * kets_;

  //--------------------Here Parallel mat mult------------------------//
#pragma omp parallel 
  {
    int id, l_start, l_end, Nthrds;
    Nthrds  = omp_get_num_threads();
    id      = omp_get_thread_num();
    l_start = id * size / Nthrds;
    l_end   = ( id + 1 ) * size / Nthrds;

 
    if (id == Nthrds-1)
      l_end = size;
    
    int l_block_size = l_end-l_start;    

    Eigen::Matrix<type, -1, -1> tmp(s1_length, s2_length);
    tmp.setZero();

    tmp = bras_.block(l_start, 0, l_block_size, s2_length).adjoint() * kets_.block(l_start, 0, l_block_size, s1_length);


  //Here, matrix multiplication result is updated on the global variable:	      

#pragma omp critical
  {
    for(int j = 0; j < s1_length; j++)
      for(int i = 0; i < s2_length; i++)               
	mu_( s1 * SEC_M + i, s2 * SEC_M + j ) += tmp( i, j );
  }
  //------------------------------------------------------------------//

  }

  Eigen::setNbThreads(Nthrds_backup);
};  




void Kubo_solver_traditional::polynomial_cycle_bra(int m_start,  int m_end){

  for( int m = m_start, i=0; m < m_end; m++, i++ ){
    if( m == 0 )
      device_.traceover(bras_.col(0).data(), pp_vec_2_, 0, 1);
    else if( m == 1 ){
      device_.H_ket ( p_vec_2_, pp_vec_2_, dmp_op_, dis_vec_);
      device_.traceover(bras_.col(1).data(), p_vec_2_, 0, 1);
    }
    else{
      device_.update_cheb( vec_2_, p_vec_2_, pp_vec_2_, dmp_op_, dis_vec_);
      device_.traceover(bras_.col(i).data(), vec_2_, 0, 1);
    }
  }
}



void Kubo_solver_traditional::polynomial_cycle_ket(int m_start,  int m_end){

  for( int m = m_start, i=0; m < m_end; m++, i++ ){
    if( m == 0 ){
      device_.vel_op( tmp_, pp_vec_ );
      device_.traceover(kets_.col(0).data(), tmp_, 0, 1);
    }
    else if( m == 1 ){
      device_.H_ket ( p_vec_, pp_vec_, dmp_op_, dis_vec_);
      device_.vel_op( tmp_, p_vec_ );
      device_.traceover(kets_.col(1).data(), tmp_, 0, 1);
    }
    else{
      device_.update_cheb( vec_, p_vec_, pp_vec_, dmp_op_, dis_vec_);
      device_.vel_op( tmp_, vec_ );
      device_.traceover(kets_.col(i).data(), tmp_, 0, 1);
    }
  }
}


