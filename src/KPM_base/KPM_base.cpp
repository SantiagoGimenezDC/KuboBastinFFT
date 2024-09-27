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

 

#include "../complex_op.hpp"
#include "KPM_base.hpp"

#include "../Kubo_solver/time_station_2.hpp"


void KPM_base::initialize_device(){

  device_.build_Hamiltonian();
  device_.setup_velOp();
  
  if(parameters_.a_ == 1.0){
    r_type Emin, Emax;
    device_.minMax_EigenValues(300, Emax,Emin);

    
    parameters_.a_ =  ( Emax - Emin ) / ( 2.0 - parameters_.edge_ );
    parameters_.b_ = -( Emax + Emin ) / 2.0;

  }
  
  device_.adimensionalize( parameters_.a_, parameters_.b_ );
}



KPM_base::KPM_base(solver_vars& parameters, Device& device) : parameters_(parameters), device_(device)
{

  if(parameters_.cap_choice_ == 0)
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else if(parameters_.cap_choice_==1)
    cap_      = new Effective_Contact(parameters_.E_min_, parameters_.eta_/parameters_.a_);

  
  if(parameters_.base_choice_ == 0)
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 1 )
    vec_base_ = new Complex_Phase(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 2 )
    vec_base_ = new Complex_Phase_real(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 3 )
    vec_base_ = new FullTrace(device_.parameters(), parameters_.seed_);

  
  if(parameters_.kernel_choice_ == 0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();
  
}





KPM_base::~KPM_base(){

/*------------Delete everything--------------*/

  delete []rand_vec_;
  delete []dmp_op_;
   
  delete kernel_;
  delete cap_;
  delete vec_base_;
}


void KPM_base::compute(){

  time_station_2 solver_station;
  solver_station.start();
  

  
  //----------------Initializing the Device---------------//
  //------------------------------------------------------//
  time_station_2 hamiltonian_setup_time;
  hamiltonian_setup_time.start();

  initialize_device();  

  hamiltonian_setup_time.stop("    Time to setup the Hamiltonian:            ");
  std::cout<<std::endl;

  //------------------------------------------------------//
  //------------------------------------------------------//  




  //-------------------Allocating memory------------------//
  //------------------------------------------------------//  
  time_station_2 allocation_time;
  allocation_time.start();

  allocate_memory();
  
  allocation_time.stop("\n \nAllocation time:            ");
  //------------------------------------------------------//
  //------------------------------------------------------//  



  
  //-------------------This shouldnt be heeere--------------//
  int W      = device().parameters().W_,
      C      = device().parameters().C_,
      LE     = device().parameters().LE_;

  r_type a       = parameters_.a_,
         E_min   = parameters_.E_min_,
         eta     = parameters_.eta_;
    
  E_min /= a;
  eta   /= a;

  
  cap().create_CAP(W, C, LE,  dmp_op_);
  device().damp(dmp_op_);
  //-------------------This shouldnt be heeere--------------//  


  
  int R = parameters().R_,
      D = parameters().dis_real_;



  
  for(int d = 1; d <= D; d++){


    time_station_2 total_csrmv_time;
    time_station_2 total_FFTs_time;
	
    //device_.Anderson_disorder(dis_vec_);
    device_.update_dis( dmp_op_);




    
    for(int r = 1; r <= R; r++){


      time_station_2 randVec_time;
      randVec_time.start();
      
      std::cout<<std::endl<< std::to_string( ( d - 1 ) * R + r)+"/"+std::to_string( D * R )+"-Vector/disorder realization;"<<std::endl;

      vec_base_->generate_vec_im( rand_vec_, r);       
      device_.rearrange_initial_vec( rand_vec_ ); //very hacky
      

      rand_vec_iteration();
    
  }


  
  solver_station.stop("Total case execution time:              ");
  }
};






void KPM_DOS_solver::allocate_memory(){
  Chebyshev_states<State<type>> cheb_vectors( device() );
}



void KPM_DOS_solver::rand_vec_iteration(){
  Chebyshev_states<State<type>> cheb_vectors( device() );
  int M   = parameters_.M_;


  State<type> init_state( device().parameters().DIM_, rand_vec() );
  cheb_vectors.reset( init_state );
  

//=================================KPM Step 0======================================//
  //moments(0)=v0**cheb_vectors(0);
  
//=================================KPM Step 1======================================//       

  cheb_vectors.update();
  //moments(1)=v0*cheb_vectors(1);
  
//=================================KPM Steps 2 and on===============================//
    
  for( int m = 2; m < M; m++ ){
    cheb_vectors.update();
    //moments(m)=v0*cheb_vectors(2).data()
  }

}
