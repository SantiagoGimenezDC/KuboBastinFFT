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


#include "Kubo_solver_filtered.hpp"

#include "../time_station.hpp"
#include "../time_station_2.hpp"
#include "../../Device/Device.hpp"
#include "../../Device/Graphene_KaneMele.hpp"


void Kubo_solver_filtered::compute_imag(){

  time_station_2 solver_station;
  solver_station.start();



  //----------------Initializing the Device---------------//
  time_station_2 hamiltonian_setup_time;
  hamiltonian_setup_time.start();

  
  device_.build_Hamiltonian();


  device_.setup_velOp();
  
  if(parameters_.a_ == 1.0){
    r_type Emin = 0, Emax = 0;
    device_.minMax_EigenValues(300, Emax,Emin);

    parameters_.a_ = ( Emax - Emin ) / ( 2.0 - parameters_.edge_ );
    parameters_.b_ = - ( Emax + Emin ) / 2.0;

  }

  hamiltonian_setup_time.stop("    Time to setup the Hamiltonian:            ");
  std::cout<<std::endl; 
  //-------------Finish Initializing the Device-----------//



  //----------Initialize filter and filter variables    
  filter_.compute_filter();   
  filter_.compute_k_dis(parameters_.a_,parameters_.b_);
  device_.adimensionalize(parameters_.a_, parameters_.b_);
  //----------Finish initializing the filter   

  


  //--------------------Catch all necessary parameters--------------------//
  int W      = device_.parameters().W_,
      C      = device_.parameters().C_,
      LE     = device_.parameters().LE_,
      DIM    = device_.parameters().DIM_,
      SUBDIM = device_.parameters().SUBDIM_;    

  int num_parts = parameters_.num_parts_,
      SEC_SIZE    = 0;

  SEC_SIZE = SUBDIM / num_parts + SUBDIM % num_parts; // Assuming always that % is much smaller than /, hopefully. Not true if num_parts \approx SUBDIM.
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



  int M_dec = filter_.M_dec();

  parameters_.num_p_ = filter_.parameters().nump_;
  nump = parameters_.num_p_;
  //-------------Finished Catching all necessary parameters----------------//
  

  bool double_buffer = false;

  
  //---------------------------Memory estimates-----------------------// 
  r_type buffer_mem    = r_type( 2.0 * r_type(M_dec) * r_type( SEC_SIZE) * sizeof(type) ) / r_type( 1E9 ),
         recursion_mem = r_type( ( 5 * r_type( DIM ) + 1 * r_type( SUBDIM ) ) * sizeof(type) )/ r_type( 1E9 ),
         FFT_mem       = 0.0,
         Ham_mem = device_.Hamiltonian_size()/ r_type( 1E9 ),
         Total = 0.0;

  //  if(parameters_.base_choice_ == 1 )
    buffer_mem*=2;

  if(double_buffer == true )
    buffer_mem*=2;
  
  
  FFT_mem = r_type( ( 1 + omp_get_num_threads() * ( 8 + 1 ) ) * nump * sizeof(type) ) / r_type( 1E9 );
  
  Total = buffer_mem + Ham_mem + recursion_mem + FFT_mem;

  
  std::cout<<std::endl;
  std::cout<<"Expected memory cost breakdown:"<<std::endl;
  std::cout<<"   Chebyshev buffers:    "<< buffer_mem<<" GBs"<<std::endl;  
  std::cout<<"   Hamiltonian size:     "<< Ham_mem<<" GBs"<<std::endl;  
  std::cout<<"   Recursion vectors:    "<<  recursion_mem <<" GBs"<<std::endl;
  std::cout<<"   FFT auxiliary lines:  "<<  FFT_mem <<" GBs"<<std::endl<<std::endl;   
  std::cout<<"TOTAL:  "<<  Total<<" GBs"<<std::endl<<std::endl;
  //--------------------Finished Memory estimates--------------------// 

  

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



  type **d_bras_re,
       **d_kets_re,
       **d_bras_im,
       **d_kets_im;

  if( (sym_formula_ == KUBO_BASTIN || sym_formula_ == KUBO_SEA ) && double_buffer ){
    d_bras_re = new type* [ M_dec ];
    d_kets_re = new type* [ M_dec ];
    d_bras_im = new type* [ M_dec ];
    d_kets_im = new type* [ M_dec ];

    for(int m=0;m<M_dec;m++){
      d_bras_re[m] = new type [ SEC_SIZE ],
      d_kets_re[m] = new type [ SEC_SIZE ];
      d_bras_im[m] = new type [ SEC_SIZE ],
      d_kets_im[m] = new type [ SEC_SIZE ];

    }
  }

  
 
  //Recursion Vectors
  type *rand_vec = new type  [ DIM ];
  
  //Auxiliary - disorder and CAP vectors
  r_type *dmp_op  = new r_type [ DIM ],
         *dis_vec = new r_type[ SUBDIM ];
/*-----------------------------------------------*/





  
  
  
/*---------------Dataset vectors----------------*/
  type r_data      [ 2 * nump ],
       final_data  [ 2 * nump ];

  r_type conv_R      [ 2 * D * R ],
         E_points    [ nump ];

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
    E_points   [e] = 0.0;
    r_data     [e] = 0.0;
    final_data [e] = 0.0;
    r_data     [nump+e] = 0.0;
    final_data [nump+e] = 0.0;
  }

  compute_E_points(E_points);  

  
  cap_->create_CAP(W, C, LE,  dmp_op);
  device_.damp(dmp_op);

  /*-----------------------------------------------*/  
  


  
    


  
  
  for(int d = 1; d <= D; d++){

    
    
    device_.Anderson_disorder(dis_vec);
    device_.update_dis(dis_vec, dmp_op);

   


    
    for(int r=1; r<=R;r++){


      time_station_2 total_csrmv_time;
      time_station_2 total_FFTs_time;

      time_station_2 randVec_time;
      randVec_time.start();
      
      std::cout<<std::endl<< std::to_string( ( d - 1 ) * R + r)+"/"+std::to_string( D * R )+"-Vector/disorder realization;"<<std::endl;
       
      
       vec_base_->generate_vec_im( rand_vec, r);       
       device_.rearrange_initial_vec(rand_vec); //very hacky
  
            
      if(dynamic_cast<Graphene_KaneMele*>(&device_) && device_.isKspace() && parameters_.base_choice_ == 0)
	device_.Uk_ket(rand_vec, rand_vec);

      if(dynamic_cast<Graphene_KaneMele*>(&device_) && !device_.isKspace() && parameters_.base_choice_ == 4)	
      	device_.to_kSpace(rand_vec, rand_vec, 1);

    
       for(int k=0; k<nump; k++ ){
         r_data    [ k ] = 0;
	 r_data    [ nump + k ] = 0;
       }


       
       for(int s = 0; s < num_parts; s++){

         std::cout<< "    -Part: "<<s+1<<"/"<<num_parts<<std::endl;

	 
         reset_buffer(bras_re);
         reset_buffer(bras_im);
         reset_buffer(kets_re);
         reset_buffer(kets_im);

         if( ( sym_formula_ == KUBO_BASTIN || sym_formula_ == KUBO_SEA ) && double_buffer){
           reset_buffer(d_bras_re);
           reset_buffer(d_kets_re);
           reset_buffer(d_bras_im);
           reset_buffer(d_kets_im);
         }
      

	 
	 
         time_station_2 csrmv_time_bras;
         csrmv_time_bras.start();

	 if( ( sym_formula_ == KUBO_BASTIN || sym_formula_ == KUBO_SEA ) && double_buffer)	   
           filtered_polynomial_cycle_direct_doubleBuffer_imag(bras_re, bras_im, d_bras_re, d_bras_im, rand_vec, s, 0);     
	 else
	   filtered_polynomial_cycle_direct_imag(bras_re, bras_im, rand_vec, s, 0);     
	   //filtered_polynomial_cycle_OTF_imag(bras_re, bras_im, rand_vec, s, 0);

	 
	 csrmv_time_bras.stop("           Bras cycle time:            ");
         total_csrmv_time += csrmv_time_bras;



	 

	 
	 time_station_2 csrmv_time_kets;
         csrmv_time_kets.start();

	 if( ( sym_formula_ == KUBO_BASTIN || sym_formula_ == KUBO_SEA )  && double_buffer)	   
           filtered_polynomial_cycle_direct_doubleBuffer_imag(kets_re, kets_im, d_kets_re, d_kets_im, rand_vec, s, 1);     
	 else
	   filtered_polynomial_cycle_direct_imag(kets_re, kets_im, rand_vec, s, 1);     
	   //filtered_polynomial_cycle_OTF_imag(kets_re, kets_im, rand_vec, s, 1);

	 csrmv_time_kets.stop("           Kets cycle time:            ");
         total_csrmv_time += csrmv_time_kets;
	 


	 

	 time_station_2 FFTs_time;
	 FFTs_time.start();
	
	 if(sym_formula_ == KUBO_GREENWOOD)
	   Greenwood_FFTs_imag(bras_re, bras_im, kets_re, kets_im, r_data, s);

	 else if(sym_formula_ == KUBO_BASTIN){
           if(double_buffer)
	     Bastin_FFTs_doubleBuffer_imag(E_points, bras_re, bras_im, d_bras_re, d_bras_im, kets_re, kets_im, d_kets_re, d_kets_im, r_data, 1);
	   else
	     Bastin_FFTs_imag(E_points, bras_re, bras_im, kets_re, kets_im, r_data, 1);
	 }
	 if(sym_formula_ == KUBO_SEA){
	   if(double_buffer)
	     Sea_FFTs_doubleBuffer_imag(E_points, bras_re, bras_im, d_bras_re, d_bras_im, kets_re, kets_im, d_kets_re, d_kets_im, r_data, 1);
           else
	     Sea_FFTs_imag(E_points, bras_re, bras_im, kets_re, kets_im, r_data, 1);
	 }
	 
	 FFTs_time.stop("           FFT operations time:        ");
	 total_FFTs_time += FFTs_time;
 



	 
       }


      total_csrmv_time.print_time_msg( "\n       Total CSRMV time:           ");
      total_FFTs_time.print_time_msg("       Total FFTs time:            ");





      
      
      time_station_2 time_postProcess;
      time_postProcess.start();

      if(sym_formula_ == KUBO_GREENWOOD)
        update_data(E_points,  r_data, final_data, conv_R, ( d - 1 ) * R + r, run_dir, filename);
       
      if(sym_formula_ == KUBO_BASTIN)
        update_data_Bastin(E_points, r_data, final_data, conv_R, ( d - 1 ) * R + r, run_dir, filename);

      if(sym_formula_ == KUBO_SEA)
        update_data_Sea(E_points, r_data, final_data, conv_R, ( d - 1 ) * R + r, run_dir, filename);
      
	 

      time_postProcess.stop( "       Post-processing time:       ");



      
      
      randVec_time.stop("       Total RandVec time:         ");
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

  
  if(sym_formula_ == KUBO_BASTIN && double_buffer){
    for(int m=0;m<M_dec;m++){
      delete []d_bras_re[m];
      delete []d_kets_re[m];
      delete []d_bras_im[m];
      delete []d_kets_im[m];
    }
  
    delete []d_bras_re;
    delete []d_kets_re;
    delete []d_bras_im;
    delete []d_kets_im;
  }  
  
  delete []rand_vec;
  delete []dmp_op;
  delete []dis_vec;
  
  /*-----------------------------------------------*/
  
  solver_station.stop("Total case execution time:              ");
}





void Kubo_solver_filtered::filter_imag( int m, type* new_vec, type** poly_buffer_re, type** poly_buffer_im, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ,
      size = SEC_SIZE;

  std::vector<int> list = filter_.decimated_list();
  int M_dec = list.size();
  
  bool cyclic = true;
  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2;

  
  r_type KB_window[L];
 
  for(int i=0; i < L; i++)
    KB_window[i] = filter_.KB_window()[i];


  if( s != parameters_.num_parts_ - 1 )
    size -= device_.parameters().SUBDIM_ % parameters_.num_parts_;
  
  
  
  type factor = ( 2 - ( m == 0 ) ) * kernel_->term(m,M) * std::polar( 1.0,  M_PI * m * (  - 2 * k_dis + initial_disp_ ) / M_ext );

  
  if( vel_op == 1 ){
    device_.vel_op( tmp_velOp, new_vec, parameters_.vel_dir_2_ );
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

    if( dist < Np || ( Np == 0 && dist == 0 ))
      plus_eq_imag( poly_buffer_re[ i ], poly_buffer_im[ i ], tmp,  factor * KB_window[ Np + dist  ], size);
    
    }
  
};


void Kubo_solver_filtered::filter_doubleBuffer_imag( int m, type* new_vec, type** poly_buffer_re, type** poly_buffer_im, type** d_poly_buffer_re, type** d_poly_buffer_im, type* tmp, type* tmp_velOp, int s, int vel_op ){
  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ,
      size = SEC_SIZE;

  std::vector<int> list = filter_.decimated_list();
  int M_dec = list.size();
  
  bool cyclic = true;
  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2;

  
  r_type KB_window[L];
 
  for(int i=0; i < L; i++)
    KB_window[i] = filter_.KB_window()[i];


  if( s != parameters_.num_parts_ - 1 )
    size -= device_.parameters().SUBDIM_ % parameters_.num_parts_;
  
  
  
  type factor = ( 2 - ( m == 0 ) ) * kernel_->term(m,M) * std::polar( 1.0,  M_PI * m * (  - 2 * k_dis + initial_disp_ ) / M_ext );

  
  if( vel_op == 1 ){
    device_.vel_op( tmp_velOp, new_vec, parameters_.vel_dir_2_ );
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

    if( ( dist < Np ) || ( Np == 0 && dist == 0 ) ){
      plus_eq_imag( poly_buffer_re[ i ], poly_buffer_im[ i ], tmp,  factor * KB_window[ Np + dist  ], size);
      plus_eq_imag( d_poly_buffer_re[ i ], d_poly_buffer_im[ i ], tmp,  m * factor * KB_window[ Np + dist  ], size);
    }
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
    device_.vel_op( pp_vec, rand_vec, parameters_.vel_dir_1_ );  
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
    /*
      if( m == M-1)
	for(int i=0; i<filter_.M_dec();i++)
	std::cout<<i<<"/"<<filter_.M_dec()<<"   "<<poly_buffer_re[i][SEC_SIZE/2]<<std::endl;
    */
  }
      
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}


void Kubo_solver_filtered::filtered_polynomial_cycle_direct_doubleBuffer_imag(type** poly_buffer_re, type** poly_buffer_im, type** d_poly_buffer_re, type** d_poly_buffer_im, type rand_vec[],  int s, int vel_op){
  
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
    device_.vel_op( pp_vec, rand_vec, parameters_.vel_dir_1_ );  
  else
#pragma omp parallel for
    for(int l = 0; l < DIM; l++)
      pp_vec[l] = - rand_vec[l]; //This minus sign is due to the CONJUGATION of applying both velocity operators to the KET side!!!!

  
 
  filter_doubleBuffer_imag( 0, pp_vec, poly_buffer_re, poly_buffer_im, d_poly_buffer_re, d_poly_buffer_im, tmp, tmp_velOp, s, vel_op );  


  
  device_.H_ket ( p_vec, pp_vec );
  filter_doubleBuffer_imag( 1, p_vec, poly_buffer_re, poly_buffer_im,  d_poly_buffer_re, d_poly_buffer_im, tmp, tmp_velOp, s, vel_op ); 


  for( int m=2; m<M; m++ ){

    device_.update_cheb( vec, p_vec, pp_vec );

    filter_doubleBuffer_imag( m, vec, poly_buffer_re, poly_buffer_im,  d_poly_buffer_re, d_poly_buffer_im, tmp, tmp_velOp, s, vel_op );
    
  }
      
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}





