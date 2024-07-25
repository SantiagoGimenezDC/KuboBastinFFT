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


#include "../static_vars.hpp"

#include "../vec_base.hpp"
#include "../CAP.hpp"
#include "../kernel.hpp"
#include "../Device/Device.hpp"
#include "../Device/Graphene.hpp"

#include "../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"

#include "time_station.hpp"



void Kubo_solver_filtered::compute_imag(){
  
  auto start0 = std::chrono::steady_clock::now();

  //----------------Initializing the Device---------------//
  time_station device_init_time;
  time_station hamiltonian_setup_time;
  
  device_.build_Hamiltonian();

  hamiltonian_setup_time.stop("    Time to setup the Hamiltonian:            ");
  


  device_.setup_velOp();
  
  if(parameters_.a_ == 1.0){
    r_type Emin = 0, Emax = 0;
    device_.minMax_EigenValues(300, Emax,Emin);

    parameters_.a_ = ( Emax - Emin ) / ( 2.0 - parameters_.edge_ );
    parameters_.b_ = - ( Emax + Emin ) / 2.0;

  }


  
  filter_.compute_k_dis(parameters_.a_,parameters_.b_);

  device_.adimensionalize(parameters_.a_, parameters_.b_);

  device_init_time.stop("    Time to setup the whole device:         ");
  std::cout<<std::endl;
  

  int W      = device_.parameters().W_,
      C      = device_.parameters().C_,
      LE     = device_.parameters().LE_,
      DIM    = device_.parameters().DIM_,
      SUBDIM = device_.parameters().SUBDIM_;    

  int num_parts = parameters_.num_parts_,
      SEC_SIZE    = 0;

  SEC_SIZE = SUBDIM / num_parts;
  parameters_.SECTION_SIZE_ = SEC_SIZE;

  
  r_type a       = parameters_.a_,
         E_min   = parameters_.E_min_,
    //   E_start = parameters_.E_start_,
    //   E_end   = parameters_.E_end_,
         eta     = parameters_.eta_;
  
  int R = parameters_.R_,
      D = parameters_.dis_real_,
      nump;
 
  std::string run_dir  = parameters_.run_dir_,
              filename = parameters_.filename_;

  E_min /= a;
  eta   /= a;

  filter_.compute_filter(); //Initialize filter and filter variables

  int M_dec = filter_.M_dec();

  parameters_.num_p_ = filter_.parameters().nump_;
  nump = parameters_.num_p_;
  
  auto start_BT = std::chrono::steady_clock::now();




/*------------Big memory allocation--------------*/
  //Single Shot vectors
  type **bras_re = new type* [ M_dec ],
       **kets_re = new type* [ M_dec ],
       **bras_im = new type* [ M_dec ],
       **kets_im = new type* [ M_dec ];

  for(int m=0;m<M_dec;m++){
    bras_re[m] = new type [ SEC_SIZE ],
    kets_re[m] = new type [ SEC_SIZE ];
    bras_im[m] = new type [ SEC_SIZE ],
    kets_im[m] = new type [ SEC_SIZE ];
  }

 
  //Recursion Vectors
  type *rand_vec = new type  [ DIM ];
  
  //Auxiliary - disorder and CAP vectors
  r_type *dmp_op  = new r_type [ DIM ],
         *dis_vec = new r_type[ SUBDIM ];
/*-----------------------------------------------*/





  
  
  
/*---------------Dataset vectors----------------*/
  r_type r_data      [ nump ],
         final_data  [ nump ],
         conv_R      [ 2 * D * R ],
         E_points    [ nump ];

  r_type integrand   [ nump ] ;
/*-----------------------------------------------*/  


 


  
/*----------------Initializations----------------*/

      
#pragma omp parallel for  
  for(int k=0; k<DIM; k++){
    rand_vec [k] = 0.0;
    dmp_op   [k] = 1.0;
  }


  for( int r = 0; r < D * R; r++ )
    conv_R [r] = 0.0;

  
  for(int e=0; e<nump;e++){
    E_points   [e] = filter_.E_points()[e];
    r_data     [e] = 0.0;
    final_data [e] = 0.0;
    integrand  [e] = 0.0;
  }
/*-----------------------------------------------*/  

  compute_E_points(E_points);  

 
  cap_->create_CAP(W, C, LE,  dmp_op);
  device_.damp(dmp_op);
  
  

  r_type buffer_mem    = r_type( 2 * r_type(M_dec) * r_type( SEC_SIZE) * sizeof(type) ) / r_type( 1E9 ),
         recursion_mem = r_type( ( 5 * r_type( DIM ) + 1 * r_type( SUBDIM ) ) * sizeof(type) )/ r_type( 1E9 ),
         FFT_mem       = 0.0,
         Ham_mem = device_.Hamiltonian_size()/ r_type( 1E9 ),
         Total = 0.0;

  
  FFT_mem = r_type( ( 1 + omp_get_num_threads() * ( 8 + 1 ) ) * nump * sizeof(type) ) / r_type( 1E9 );
  
  Total = buffer_mem + Ham_mem + recursion_mem + FFT_mem;

  
  std::cout<<std::endl;
  std::cout<<"Expected memory cost breakdown:"<<std::endl;
  std::cout<<"   Chebyshev buffers:    "<< buffer_mem<<" GBs"<<std::endl;  
  std::cout<<"   Hamiltonian size:     "<< Ham_mem<<" GBs"<<std::endl;  
  std::cout<<"   Recursion vectors:    "<<  recursion_mem <<" GBs"<<std::endl;
  std::cout<<"   FFT auxiliary lines:  "<<  FFT_mem <<" GBs"<<std::endl<<std::endl;   
  std::cout<<"TOTAL:  "<<  Total<<" GBs"<<std::endl<<std::endl;

  
  auto end_BT = std::chrono::steady_clock::now();
  Station( std::chrono::duration_cast<std::chrono::microseconds>(end_BT - start_BT).count()/1000, "    Bloat time:            ");


  
  
  for(int d = 1; d <= D; d++){

    int total_csrmv = 0,
        total_FFTs  = 0;
    
    device_.Anderson_disorder(dis_vec);
    device_.update_dis(dis_vec, dmp_op);

   


    
    for(int r=1; r<=R;r++){
       
      auto start_RV = std::chrono::steady_clock::now();
      std::cout<<std::endl<< ( d - 1 ) * R + r <<"/"<< D * R << "-Vector/disorder realization;"<<std::endl;


      reset_buffer(bras_re);
      reset_buffer(kets_re);
      reset_buffer(bras_im);
      reset_buffer(kets_im);

      
       vec_base_->generate_vec_im( rand_vec, r);       
       device_.rearrange_initial_vec(rand_vec); //very hacky
  

    
       for(int k=0; k<nump; k++ ){
         r_data    [k] = 0;
         integrand [k] = 0;
       }


       
       for(int s=0;s<=num_parts;s++){

	 if( s==num_parts && SUBDIM % num_parts==0  )
	   break;

	 if(SUBDIM % num_parts==0)
           std::cout<< "    -Part: "<<s+1<<"/"<<num_parts<<std::endl;
         else
	   std::cout<< "    -Part: "<<s+1<<"/"<<num_parts+1<<std::endl;




         auto csrmv_start_2 = std::chrono::steady_clock::now();
    
         filtered_polynomial_cycle_direct_imag(bras_re, bras_im, rand_vec, s, 0);     

         auto csrmv_end_2 = std::chrono::steady_clock::now();
         Station(std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000, "           Bras cycle time:            ");
  
	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000;  


	 

	 
         auto csrmv_start = std::chrono::steady_clock::now();

	 filtered_polynomial_cycle_direct_imag(kets_re, kets_im, rand_vec, s, 1);
	 
         auto csrmv_end = std::chrono::steady_clock::now();
         Station( std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000, "           Kets cycle time:            ");

	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000;

	 

	 
	 
         auto FFT_start_2 = std::chrono::steady_clock::now();

	 Greenwood_FFTs_imag(bras_re, bras_im, kets_re, kets_im, r_data);
	 
	 auto FFT_end_2 = std::chrono::steady_clock::now();
         Station(std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count()/1000, "           FFT operations time:        ");
    
	 total_FFTs += std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count()/1000;

       }




  
       std::cout<<std::endl<<"       Total CSRMV time:           "<< total_csrmv<<" (ms)"<<std::endl;
       std::cout<<"       Total FFTs time:            "<< total_FFTs<<" (ms)"<<std::endl;


       
       auto start_pr = std::chrono::steady_clock::now();
    
       //integration(E_points, integrand, r_data);
    
       auto end_pr = std::chrono::steady_clock::now();
       Station(std::chrono::duration_cast<std::chrono::microseconds>(end_pr - start_pr).count()/1000, "       Integration time:           ");
       

       
    
       auto plot_start = std::chrono::steady_clock::now();    

       /*When introducing a const. eta with modified polynomials, the result is equals to that of a
       simulation with regular polynomials and an variable eta_{var}=eta*sin(acos(E)). The following
       heuristical correction greatly improves the result far from the CNP to match that of the
       desired regular polys and const. eta.*/
       if( parameters_.eta_!=0 )
         eta_CAP_correct(E_points, r_data);


       update_data(E_points, r_data, final_data, conv_R, ( d - 1 ) * R + r, run_dir, filename);
       plot_data(run_dir,filename);

       
       auto plot_end = std::chrono::steady_clock::now();
       Station(std::chrono::duration_cast<std::chrono::microseconds>(plot_end - plot_start).count()/1000, "       Plot and update time:       ");


    
       
       auto end_RV = std::chrono::steady_clock::now();    
       Station(std::chrono::duration_cast<std::chrono::milliseconds>(end_RV - start_RV).count(), "       Total RandVec time:         ");
       std::cout<<std::endl;
    }
  }

  /*------------Delete everything--------------*/
  //Single Shot vectors
  for(int m=0;m<M_dec;m++){
    delete []bras_re[m];
    delete []kets_re[m];
    delete []bras_im[m];
    delete []kets_im[m];
  }
  
  delete []bras_re;
  delete []kets_re;
  delete []bras_im;
  delete []kets_im;
  
  delete []rand_vec;
  delete []dmp_op;
  delete []dis_vec;
  
  /*-----------------------------------------------*/
  

  auto end = std::chrono::steady_clock::now();   
  Station(std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count(), "Total case execution time:             ");
  std::cout<<std::endl;
}



void Kubo_solver_filtered::filter_imag( int m, type* new_vec, type** poly_buffer_re, type** poly_buffer_im, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ;

  std::vector<int> list = filter_.decimated_list();
  int M_dec = list.size();
  
  bool cyclic = true;
  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2;

  
  r_type KB_window[L];
 
  for(int i=0; i < L; i++)
    KB_window[i] = filter_.KB_window()[i];


  
  
  type factor = ( 2 - ( m == 0 ) ) * kernel_->term(m,M) * std::polar( 1.0,  M_PI * m * (  - 2 * k_dis + initial_disp_ ) / M_ext );

  
  if( vel_op == 1 ){
    device_.vel_op( tmp_velOp, new_vec );
    device_.traceover(tmp, tmp_velOp, s, num_parts);
  }
  else
    device_.traceover(tmp, new_vec, s, num_parts);


      
  for(int i = 0; i < M_dec; i++ ){
    int dist = abs( m - list[i] );

    if( cyclic ){
      if( ( list[i] < Np && m > M_ext - Np - 1 ) )
        dist = M_ext - m + list[i];
      if( ( m < Np && list[i] > M_ext - Np - 1 ) )
        dist = M_ext - list[i] + m ;
    }

    if( dist < Np )
      plus_eq_imag( poly_buffer_re[ i ], poly_buffer_im[ i ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
    
  }
  
};



void Kubo_solver_filtered::filtered_polynomial_cycle_direct_imag(type** poly_buffer_re, type** poly_buffer_im, type rand_vec[],  int s, int vel_op){
  
  int M         = parameters_.M_,
      DIM       = device_.parameters().DIM_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ;


    
  type *vec      = new type [ DIM ],
       *p_vec    = new type [ DIM ],
       *pp_vec   = new type [ DIM ],
       *tmp      = new type [ SEC_SIZE ];

  type *tmp_velOp = new type [ DIM ];
  

  
#pragma omp parallel for
  for(int l=0;l<DIM;l++){
    vec[l] = 0;
    p_vec[l] = 0;
    pp_vec[l] = 0;

    tmp_velOp[l] = 0;

    if( l < SEC_SIZE )
      tmp[l] = 0;
  }


  
  if( vel_op == 1 )
    device_.vel_op( pp_vec, rand_vec );  
  else
#pragma omp parallel for
    for(int l = 0; l < DIM; l++)
      pp_vec[l] = - rand_vec[l]; //This minus sign is due to the CONJUGATION of applying both velocity operators to the KET side!!!!

  
 
  filter_imag( 0, pp_vec, poly_buffer_re, poly_buffer_im, tmp, tmp_velOp, s, vel_op );  


  
  device_.H_ket ( p_vec, pp_vec );
  filter_imag( 1, p_vec, poly_buffer_re, poly_buffer_im, tmp, tmp_velOp, s, vel_op ); 


  for( int m=2; m<M; m++ ){

    device_.update_cheb( vec, p_vec, pp_vec );

    filter_imag( m, vec, poly_buffer_re, poly_buffer_im, tmp, tmp_velOp, s, vel_op );
  }
      
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}





