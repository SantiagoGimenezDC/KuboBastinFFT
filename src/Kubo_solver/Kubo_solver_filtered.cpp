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



Kubo_solver_filtered::Kubo_solver_filtered(solver_vars& parameters, Device& device, KB_filter& filter) : parameters_(parameters), device_(device), filter_(filter)
{

  if(parameters_.cap_choice_==0)
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else if(parameters_.cap_choice_==1)
    cap_      = new Effective_Contact(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);


  
  if(parameters_.base_choice_ == 0 )
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 1 )
    vec_base_ = new Complex_Phase(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_ == 2 )
    vec_base_ = new Complex_Phase_real(device_.parameters(), parameters_.seed_);
  else
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);

  if(parameters_.kernel_choice_==0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();
  else
    kernel_   = new None();
}
/*
void Station(int millisec, std::string msg ){
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<msg;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl;

}
*/

void Kubo_solver_filtered::reset_buffer(type** polys){
    
  int M_dec = filter_.M_dec(),
    SEC_SIZE    =   parameters_.SECTION_SIZE_;

#pragma omp parallel for
  for(int m = 0; m < M_dec; m++)
    for(int i = 0; i < SEC_SIZE; i++)
      polys[m][i] = 0.0;
};

void Kubo_solver_filtered::compute_E_points( r_type* E_points ){
      
  int k_dis = filter_.parameters().k_dis_,
      M_ext = filter_.parameters().M_ext_,
      nump = filter_.parameters().nump_;

  //  for(int k = -nump/2; k < nump ; k++)
  //	E_points[ k +nump/2]            = cos( 2 * M_PI * ( k - k_dis + 0.25 ) / M_ext );             
	
    
  
    if( nump % 2 == 1 ){
       for(int k = 0; k < nump/2 ; k++){
	 E_points[ k ]                = cos( 2 * M_PI * ( k - k_dis + 0.25 ) / M_ext );             
	 E_points[ nump / 2 + k + 1 ] = cos( 2 * M_PI * ( nump / 2 - ( k - k_dis + 0.25 ) )  / M_ext );
       }
       E_points[ nump / 2 ]           = cos( 2 * M_PI * ( nump / 2 - ( - k_dis + 0.25 ) )  / M_ext );             
    }
    else
      for(int k = 0; k < nump/2 ; k++){
	E_points[ k ]            = cos( 2 * M_PI * ( k - k_dis + 0.25 ) / M_ext );             
	E_points[ nump / 2 + k ] = cos( 2 * M_PI * ( nump / 2 - ( k - k_dis + 0.25 ) )  / M_ext );
      }
  

};



void Kubo_solver_filtered::compute(){
  
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
  type **bras = new type* [ M_dec ],
       **kets = new type* [ M_dec ];

  for(int m=0;m<M_dec;m++){
    bras[m] = new type [ SEC_SIZE ],
    kets[m] = new type [ SEC_SIZE ];
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
  for(int m = 0; m < M_dec; m++)
    for(int l = 0; l < SEC_SIZE; l++){    
      bras [m][l] = 0.0;
      kets [m][l] = 0.0;
    }
  
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


      reset_buffer(bras);
      reset_buffer(kets);

      
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
    
         filtered_polynomial_cycle_direct_2(bras, rand_vec, s, 0);     

         auto csrmv_end_2 = std::chrono::steady_clock::now();
         Station(std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000, "           Bras cycle time:            ");
  
	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000;  


	 

	 
         auto csrmv_start = std::chrono::steady_clock::now();

	 filtered_polynomial_cycle_direct_2(kets, rand_vec, s, 1);
	 
         auto csrmv_end = std::chrono::steady_clock::now();
         Station( std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000, "           Kets cycle time:            ");

	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000;

	 

	 
	 
         auto FFT_start_2 = std::chrono::steady_clock::now();

	 Greenwood_FFTs(bras, kets, r_data);
	 
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


       update_data(E_points, integrand, r_data, final_data, conv_R, ( d - 1 ) * R + r, run_dir, filename);
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
    delete []bras[m];
    delete []kets[m];
  }
  
  delete []bras;
  delete []kets;
  delete []rand_vec;
  delete []dmp_op;
  delete []dis_vec;
  
  /*-----------------------------------------------*/
  

  auto end = std::chrono::steady_clock::now();   
  Station(std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count(), "Total case execution time:             ");
  std::cout<<std::endl;
}




inline
void copy_vector(type vec_destination[], type vec_original[], int size){
#pragma omp parallel for
  for(int i=0;i<size;i++)
    vec_destination[i] = vec_original[i];
}

inline
void plus_eq(type vec_1[], type vec_2[], type factor, int size){
#pragma omp parallel for
  for(int i=0;i<size;i++)
    vec_1[i] += factor * vec_2[i];
}

inline
void ay(type factor, type vec[], int size){
#pragma omp parallel for
  for(int i=0;i<size;i++)
    vec[i] * factor;
}



void Kubo_solver_filtered::filter( int m, type* new_vec, type** poly_buffer, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ;

  
  bool cyclic = true;
  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2,
      decRate   = filter_.parameters().decRate_;

  
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


      
  for(int dist = -Np; dist <= Np;  dist++){
    int i = ( m + dist );  

    if( i >= 0 && i < M - 1 && i % decRate == 0 ){
      int i_dec = i / decRate;  
      plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
    }
	
    if( cyclic && i < 0 && ( M - 1 + ( i + 1 )  ) % decRate == 0 )
      plus_eq( poly_buffer[ ( M - 1 + ( i + 1 ) ) / decRate ], tmp,  factor * KB_window[ Np + dist ], SEC_SIZE );
	
	
    if( cyclic && i > M - 1 && ( ( i - 1 ) - ( M - 1 )  ) % decRate == 0 )
      plus_eq( poly_buffer[ ( i - M + 1 ) / decRate ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
  }
  
};



void Kubo_solver_filtered::filter_2( int m, type* new_vec, type** poly_buffer, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ;

  std::vector<int> list = filter_.decimated_list();
  int M_dec = list.size();
  
  bool cyclic = true;
  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2,
      decRate   = filter_.parameters().decRate_;

  
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

    if( cyclic )
      if( ( list[i] < Np && m > M - Np - 1 ) || ( m < Np && list[i] > M - Np - 1 ) )
        dist = M - ( m + list[i] );
    
    if(dist < Np)
      plus_eq( poly_buffer[ i ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
  }
  
};



void Kubo_solver_filtered::filtered_polynomial_cycle_direct_2(type** poly_buffer, type rand_vec[],  int s, int vel_op){
  
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

  
 
  filter_2( 0, pp_vec, poly_buffer, tmp, tmp_velOp, s, vel_op );  


  
  device_.H_ket ( p_vec, pp_vec );
  filter_2( 1, p_vec, poly_buffer, tmp, tmp_velOp, s, vel_op ); 


  for( int m=2; m<M; m++ ){

    device_.update_cheb( vec, p_vec, pp_vec );

    filter_2( m, vec, poly_buffer, tmp, tmp_velOp, s, vel_op );
  }
      
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}






void Kubo_solver_filtered::filtered_polynomial_cycle(type** poly_buffer, type rand_vec[], r_type damp_op[], r_type dis_vec[], int s, int vel_op){
  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      DIM       = device_.parameters().DIM_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ;


  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2,
      decRate   = filter_.parameters().decRate_,  
      M_dec     = filter_.M_dec(),
      Np_dec    = Np / decRate;


  
  type disp_factor = std::polar(1.0,   M_PI * ( -2 * k_dis + initial_disp_ )  / M_ext );
  
  r_type KB_window[L];

  bool cyclic = true;

  
  
  for(int i=0; i < L; i++)
    KB_window[i] = filter_.KB_window()[i];


  
  type *vec_f    = new type [ DIM ],
       *p_vec_f  = new type [ DIM ],
       *pp_vec_f = new type [ DIM ],
       *vec      = new type [ DIM ],
       *p_vec    = new type [ DIM ],
       *pp_vec   = new type [ DIM ],
       *tmp      = new type [ SEC_SIZE ];

  type *tmp_velOp = new type [ DIM ];
  

  
#pragma omp parallel for
  for(int l=0;l<DIM;l++){
    vec[l] = 0;
    p_vec[l] = 0;
    pp_vec[l] = 0;

    vec_f[l] = 0; 
    p_vec_f[l] = 0;
    pp_vec_f[l] = 0;

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

  
  
  //=================================KPM Step 0======================================//

  if( vel_op == 1 ){
    device_.vel_op( tmp_velOp, pp_vec );
    device_.traceover(tmp, tmp_velOp, s, num_parts);
  }
  else
    device_.traceover(tmp, pp_vec, s, num_parts);


  
  //Filter boundary conditions
  for(int  i_dec = 0; i_dec <= Np_dec ; i_dec++ ){
    plus_eq( poly_buffer[ i_dec ], tmp, KB_window[ Np - i_dec * decRate ], SEC_SIZE );

    int dist_cyclic = ( i_dec * decRate  + ( M - 1 ) % decRate + 1 );
    if(cyclic && dist_cyclic <= Np )
      plus_eq( poly_buffer[  M_dec - 1 - i_dec ], tmp,  KB_window[ Np + dist_cyclic ], SEC_SIZE );
  }
  


  

  
  //=================================KPM Step 1======================================//     
    
  device_.H_ket ( p_vec, pp_vec, damp_op, dis_vec);
  
    
  type factor = std::polar(1.0,  M_PI * 1 * (  - 2 * k_dis + initial_disp_ ) / M_ext );


  //Building first filtered recursion vector
  plus_eq(pp_vec_f, p_vec, 2 * factor * KB_window[0], DIM);

  
  if( vel_op == 1 ){
    device_.vel_op( tmp_velOp, p_vec );
    device_.traceover(tmp, tmp_velOp, s, num_parts);
  }
  else  
    device_.traceover(tmp, p_vec, s, num_parts);


  
  //Filter boundary conditions
  for(int i_dec = 0; i_dec <= Np_dec;  i_dec++){
    plus_eq( poly_buffer[ i_dec ], tmp, 2 * factor * KB_window[Np + 1 - i_dec * decRate ], SEC_SIZE );

    int dist_cyclic = ( 1 + i_dec * decRate  + ( M - 1 ) % decRate + 1 );
    if( cyclic && dist_cyclic < Np )
      plus_eq( poly_buffer[ M_dec - 1 - i_dec ], tmp, 2 * factor * KB_window[ Np + dist_cyclic ], SEC_SIZE );
  }



  

  
  //=================================KPM Steps 2 and so on===============================//

    for( int m=2; m<M; m++ ){

      
      device_.update_cheb( vec, p_vec, pp_vec, damp_op, dis_vec);

      
      factor = 2 * std::polar(1.0,  M_PI * m * (  - 2 * k_dis + initial_disp_) / M_ext );

      if( vel_op == 1 ){
          device_.vel_op( tmp_velOp, vec );
	  device_.traceover(tmp, tmp_velOp, s, num_parts);
      }
      else
	  device_.traceover(tmp, vec, s, num_parts);

	
      //===================================Filter Boundary conditions====================// 
      if( m < L ){	
	for(int i_dec = 0; i_dec <= Np_dec ;  i_dec++ ){
          if( m - i_dec * decRate   <= Np ){
	    plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[Np + m - i_dec * decRate ], SEC_SIZE );
	  }
	  int dist_cyclic = ( m + i_dec * decRate  + ( M - 1 ) % decRate + 1 );
	  if( cyclic && dist_cyclic  <= Np ){
	    plus_eq( poly_buffer[ M_dec - 1 - i_dec ], tmp, factor * KB_window[ Np + dist_cyclic ], SEC_SIZE );
	  }
	}
      }

      
      if( M - m - 1 < L ){
	for(int i_dec = 0; i_dec <= Np_dec;  i_dec++){
	  int dist = ( i_dec * decRate + ( M - 1 ) % decRate ) - ( M - 1 - m ) ;  
          if(   std::abs(dist) <= Np  ){
	    
	    plus_eq( poly_buffer[ M_dec - 1 - i_dec ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );

	  }
	  if( cyclic && (  ( M - 1 - m ) + i_dec * decRate + 1 ) <= Np ){
	    plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[ Np - ( ( M - 1 - m ) + i_dec * decRate + 1 )  ], SEC_SIZE );
	  }
	}	

	
      }
      //==========================================================================//



      

      
      //==========================Filtered recursion==============================//
      
      if( m < L + 1 )
        plus_eq( pp_vec_f, vec,  factor * KB_window[ Np + ( m - ( Np + 1 ) ) ], DIM);

      if( m < L + 2  &&  m > 1)
        plus_eq( p_vec_f,  vec,  factor * KB_window[ Np + ( m - ( Np + 2 ) ) ], DIM);


      

      
                                   //EXCEPTIONAL CASES
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
      if( (Np + 1 ) % decRate == 0 && m == L + 2){
	  if( vel_op == 1 ){
            device_.vel_op( tmp_velOp, pp_vec_f );
            device_.traceover( poly_buffer[ ( Np + 1 ) / decRate  ], tmp_velOp, s, num_parts );
	  }
          else	
	    device_.traceover( poly_buffer[ ( Np + 1 ) / decRate ], pp_vec_f, s, num_parts );
	
	
	  if( (Np + 2) % decRate == 0 ){
	    if( vel_op == 1 ){
              device_.vel_op( tmp_velOp, p_vec_f );
              device_.traceover( poly_buffer[ ( Np + 2 ) / decRate ], tmp_velOp, s, num_parts );
	    }
            else	
	      device_.traceover( poly_buffer[ ( Np + 2 ) / decRate ], p_vec_f, s, num_parts );
	  }
      }
      if( (Np + 2) % decRate == 0 && (Np + 1) % decRate > 0 && m == L + 2 ){
	  if( vel_op == 1 ){
            device_.vel_op( tmp_velOp, p_vec_f );
            device_.traceover( poly_buffer[ Np / decRate + 1 ], tmp_velOp, s, num_parts );
	  }
          else	
	    device_.traceover( poly_buffer[ Np / decRate + 1 ], p_vec_f, s, num_parts );
      }
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

      
      

      //Filtered recursion starts at m = Np, ends at M - Np
      
      if( m > L + 1 && m < M){


	device_.update_cheb_filtered ( vec_f, p_vec_f, pp_vec_f, damp_op, dis_vec, disp_factor);

	
	if( ( m - Np ) % decRate == 0 ){
	  if( vel_op == 1 ){
            device_.vel_op( tmp_velOp, vec_f );
            device_.traceover( poly_buffer[ ( m - Np ) / decRate ], tmp_velOp, s, num_parts );
	  }
          else	
	    device_.traceover( poly_buffer[ ( m - Np ) / decRate ], vec_f, s, num_parts );
	}
      }
      //==========================================================================//
    }

    delete []vec_f;
    delete []p_vec_f;
    delete []pp_vec_f;
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}


/*
void Kubo_solver_filtered::filter( int m, type* new_vec, type** poly_buffer, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ;

  
  bool cyclic = true;
  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2,
      decRate   = filter_.parameters().decRate_;

  
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


      
  for(int dist = -Np; dist <= Np;  dist++){
    int i = ( m + dist );  

    if( i >= 0 && i < M - 1 && i % decRate == 0 ){
      int i_dec = i / decRate;  
      plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
    }
	
    if( cyclic && i < 0 && ( M - 1 + ( i + 1 )  ) % decRate == 0 )
      plus_eq( poly_buffer[ ( M - 1 + ( i + 1 ) ) / decRate ], tmp,  factor * KB_window[ Np + dist ], SEC_SIZE );
	
	
    if( cyclic && i > M - 1 && ( ( i - 1 ) - ( M - 1 )  ) % decRate == 0 )
      plus_eq( poly_buffer[ ( i - M + 1 ) / decRate ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
  }
  
};



void Kubo_solver_filtered::filtered_polynomial_cycle_direct_2(type** poly_buffer, type rand_vec[], r_type damp_op[], r_type dis_vec[], int s, int vel_op){
  
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

  
 
  filter( 0, pp_vec, poly_buffer, tmp, tmp_velOp, s, vel_op );  


  
  device_.H_ket ( p_vec, pp_vec );
  filter( 1, p_vec, poly_buffer, tmp, tmp_velOp, s, vel_op ); 


  for( int m=2; m<M; m++ ){

    device_.update_cheb( vec, p_vec, pp_vec );

    filter( m, vec, poly_buffer, tmp, tmp_velOp, s, vel_op );
  }
      
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}

*/



/*
void Kubo_solver_filtered::filtered_polynomial_cycle_direct(type** poly_buffer, type rand_vec[], r_type damp_op[], r_type dis_vec[], int s, int vel_op){
  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      DIM       = device_.parameters().DIM_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ;


  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2,
      decRate   = filter_.parameters().decRate_,  
      M_dec     = filter_.M_dec(),
      Np_dec    = Np / decRate;


  

  
  r_type KB_window[L];

  bool cyclic = true;

  
  
  for(int i=0; i < L; i++)
    KB_window[i] = filter_.KB_window()[i];


  
  type *vec_f    = new type [ DIM ],
       *p_vec_f  = new type [ DIM ],
       *pp_vec_f = new type [ DIM ],
       *vec      = new type [ DIM ],
       *p_vec    = new type [ DIM ],
       *pp_vec   = new type [ DIM ],
       *tmp      = new type [ SEC_SIZE ];

  type *tmp_velOp = new type [ DIM ];
  

  
#pragma omp parallel for
  for(int l=0;l<DIM;l++){
    vec[l] = 0;
    p_vec[l] = 0;
    pp_vec[l] = 0;

    vec_f[l] = 0; 
    p_vec_f[l] = 0;
    pp_vec_f[l] = 0;

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

  
  
  //=================================KPM Step 0======================================//

  if( vel_op == 1 ){
    device_.vel_op( tmp_velOp, pp_vec );
    device_.traceover(tmp, tmp_velOp, s, num_parts);
  }
  else
    device_.traceover(tmp, pp_vec, s, num_parts);


  
  //Filter boundary conditions
  for(int  i_dec = 0; i_dec <= Np_dec ; i_dec++ ){
    plus_eq( poly_buffer[ i_dec ], tmp, KB_window[ Np - i_dec * decRate ], SEC_SIZE );

    int dist_cyclic = ( i_dec * decRate  + ( M - 1 ) % decRate + 1 );
    if(cyclic && dist_cyclic <= Np )
      plus_eq( poly_buffer[  M_dec - 1 - i_dec ], tmp,  KB_window[ Np + dist_cyclic ], SEC_SIZE );
  }
  


  

  
  //=================================KPM Step 1======================================//     
    
  device_.H_ket ( p_vec, pp_vec, damp_op, dis_vec);
  
    
  type factor = std::polar(1.0,  M_PI * 1 * (  - 2 * k_dis + initial_disp_ ) / M_ext );


  
  if( ( Np + 1 ) % decRate == 0){
    int i = Np + 1;  
    int i_dec = i/decRate;  
    plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[ Np + Np  ], SEC_SIZE );
  }

  
  //Filter boundary conditions
  for(int i_dec = 0; i_dec <= Np_dec;  i_dec++){
    plus_eq( poly_buffer[ i_dec ], tmp, 2 * factor * KB_window[Np + 1 - i_dec * decRate ], SEC_SIZE );

    
    int dist_cyclic = ( 1 + i_dec * decRate  + ( M - 1 ) % decRate + 1 );
    if( cyclic && dist_cyclic )
      plus_eq( poly_buffer[ M_dec - 1 - i_dec ], tmp, 2 * factor * KB_window[ Np + dist_cyclic ], SEC_SIZE );
  }



  

  
  //=================================KPM Steps 2 and so on===============================//

    for( int m=2; m<M; m++ ){

      device_.update_cheb( vec, p_vec, pp_vec, damp_op, dis_vec);

      factor = 2 * std::polar(1.0,  M_PI * m * (  - 2 * k_dis + initial_disp_) / M_ext );

      if( vel_op == 1 ){
        device_.vel_op( tmp_velOp, vec );
	device_.traceover(tmp, tmp_velOp, s, num_parts);
      }
      else
	device_.traceover(tmp, vec, s, num_parts);

	
      //===================================Filter Boundary conditions====================// 
      if( m < L ){
	for(int i_dec = 0; i_dec <= Np_dec ;  i_dec++ ){
          if( m - i_dec * decRate   <= Np )
	    plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[Np + m - i_dec * decRate ], SEC_SIZE );

	  
	  int dist_cyclic = ( m + i_dec * decRate  + ( M - 1 ) % decRate + 1 );
	  if( cyclic && dist_cyclic  <= Np )
	    plus_eq( poly_buffer[ M_dec - 1 - i_dec ], tmp, factor * KB_window[ Np + dist_cyclic ], SEC_SIZE );
	  
	}
      }
      else if( M - m - 1 < L ){        
	for(int i_dec = 0; i_dec < Np_dec;  i_dec++){
	  int dist = ( i_dec * decRate + ( M - 1 ) % decRate ) - ( M - 1 - m ) ;  
          if(   std::abs(dist) <= Np  )
	    plus_eq( poly_buffer[ M_dec - 1 - i_dec ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
	  
	  if( cyclic && (  ( M - 1 - m ) + i_dec * decRate + 1 ) <= Np )
	    plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[ Np - ( ( M - 1 - m ) + i_dec * decRate + 1 )  ], SEC_SIZE );
	  
	}
      }


      
      
      for(int dist = -Np; dist <= Np;  dist++){
        int i = ( m + dist );  
	if( i % decRate == 0 && i > Np && i < M-1-Np ){
	    int i_dec = i/decRate;  
	    plus_eq( poly_buffer[ i_dec ], tmp,  factor * KB_window[ Np + dist  ], SEC_SIZE );
	}
      }
      
      //==========================================================================//
      
    }

    for(int i=0;i<M;i++)
      if(i%decRate==0)
	std::cout<<i<<std::endl;
    
    delete []vec_f;
    delete []p_vec_f;
    delete []pp_vec_f;
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}



*/






void Kubo_solver_filtered::update_data(r_type E_points[], r_type integrand[], r_type r_data[], r_type final_data[], r_type conv_R[], int r, std::string run_dir, std::string filename){

  int nump = parameters_.num_p_,
    R = parameters_.R_,
    D = parameters_.dis_real_;


  int SUBDIM = device_.parameters().SUBDIM_;    

  r_type a = parameters_.a_,
    b = parameters_.b_,
    sysSubLength = device_.sysSubLength();


  
  r_type omega = SUBDIM/( a * a * sysSubLength * sysSubLength ) / ( 2 * M_PI );//Dimensional and normalizing constant
  
  r_type tmp, max=0, av=0;

  
  //Post-processing     
  int decRate = filter_.parameters().decRate_;   
  for(int m=0;m<nump;m++)
    r_data[m] *= 2.0 * decRate * decRate  / (  1.0 - E_points[m] * E_points[m] );



  
  for(int e=0;e<nump;e++){

    tmp = final_data[e];
    final_data[e] = ( final_data [e] * (r-1.0) + omega * r_data[e] ) / r;

    if(r>1){

      tmp = std::abs( ( final_data [e] - tmp ) / tmp) ;
      if(tmp>max)
        max = tmp;

      av += tmp / nump ;
    }
  }

  if(r>1){
    conv_R[ 2 * (r-1) ]   = max;
    conv_R[ 2 * (r-1)+1 ] = av;
  }

  std::ofstream dataR;
  dataR.open(run_dir+"vecs/r"+std::to_string(r)+"_"+filename);

  
  for(int e=0;e<nump;e++)  
    dataR<< a * E_points[e] - b<<"  "<< omega * r_data [e]<<"  "<< final_data [e] <<std::endl;
    //  dataP<<  e <<"  "<< final_data [e] <<std::endl;

  
  dataR.close();



  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

    for(int e=0;e<nump;e++){    
      dataP<< a * E_points[e] - b<<"  "<< final_data [e];

      if(e<nump/2)
        dataP<<"  "<<e;
      else
        dataP<<"  "<<-(nump-e);

      dataP<<"  "<<e<<std::endl;
    }

  dataP.close();


  
  
  std::ofstream data;
  data.open(run_dir+"conv_R_"+filename);

  for(int r=1;r<D*R;r++)  
    data<< r <<"  "<< conv_R[ 2*(r-1) ]<<"  "<< conv_R[ 2*(r-1) + 1 ] <<std::endl;

  data.close();


  
  
  std::ofstream data2;
  data2.open(run_dir+"integrand_"+filename);

  for(int e=0;e<nump;e++)  
    data2<< a * E_points[e] - b<<"  "<< omega * integrand[e] <<std::endl;

  
  data2.close();
}






void Kubo_solver_filtered::plot_data(std::string run_dir, std::string filename){
        //VIEW commands

     std::string exestring=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"

         "set output '"+run_dir+filename+".png'                \n"

         "set xlabel 'E[eV]'                                               \n"
         "set ylabel  'G [2e^2/h]'                                           \n"
         
        "plot '"+run_dir+"currentResult_"+filename+"' using 1:2 w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
      char exeChar[exestring.size() + 1];
      strcpy(exeChar, exestring.c_str());    
      if(system(exeChar)){};


}










void Kubo_solver_filtered::eta_CAP_correct(r_type E_points[], r_type r_data[]){
  int nump = parameters_.num_p_;
  
  for(int e=0;e<nump;e++)
    r_data[e] *= sin(acos(E_points[e]));
}


void Kubo_solver_filtered::integration(r_type E_points[], r_type integrand[], r_type data[]){

  int M = parameters_.M_,
    nump = parameters_.num_p_;
  r_type edge = parameters_.edge_;
  
#pragma omp parallel for 
  for(int k=0; k<M-int(M*edge/4.0); k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                       //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                      //the number of points to be ignored in the future;
    for(int j=k; j<M-int(M*edge/4.0); j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      r_type ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ;
    }
  }
}
