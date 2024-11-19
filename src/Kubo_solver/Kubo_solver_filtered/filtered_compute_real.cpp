#include<omp.h>

#include "../../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"

#include "../time_station.hpp"
#include "../time_station_2.hpp"


void Kubo_solver_filtered::compute_real(){

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
      SEC_SIZE    =   parameters_.SECTION_SIZE_;




  
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

  if(parameters_.base_choice_ == 1 )
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
  type **bras = new type* [ M_dec ],
       **kets = new type* [ M_dec ];

  for(int m=0;m<M_dec;m++){
    bras[m] = new type [ SEC_SIZE ],
    kets[m] = new type [ SEC_SIZE ];
  }



  type **d_bras,
       **d_kets;

  if(sym_formula_ == KUBO_BASTIN && double_buffer ){
    d_bras = new type* [ M_dec ];
    d_kets = new type* [ M_dec ];

    for(int m=0;m<M_dec;m++){
      d_bras[m] = new type [ SEC_SIZE ],
      d_kets[m] = new type [ SEC_SIZE ];
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

    time_station_2 total_csrmv_time;
    time_station_2 total_FFTs_time;
    
    
    device_.Anderson_disorder(dis_vec);
    device_.update_dis(dis_vec, dmp_op);

   


    
    for(int r=1; r<=R;r++){

      time_station_2 randVec_time;
      randVec_time.start();
      
      std::cout<<std::endl<< std::to_string( ( d - 1 ) * R + r)+"/"+std::to_string( D * R )+"-Vector/disorder realization;"<<std::endl;
       
            
       vec_base_->generate_vec_im( rand_vec, r);       
       device_.rearrange_initial_vec(rand_vec); //very hacky


    
       for(int k=0; k<nump; k++ ){
         r_data    [ k ] = 0;
	 r_data    [ nump + k ] = 0;
       }


       
       for(int s = 0; s < num_parts; s++){

         std::cout<< "    -Part: "<<s+1<<"/"<<num_parts<<std::endl;

	 reset_buffer(bras);
         reset_buffer(kets);
      
         if(sym_formula_ == KUBO_BASTIN && double_buffer){	   
           reset_buffer(d_bras);
           reset_buffer(d_kets);
         }
      


         time_station_2 csrmv_time_kets;
         csrmv_time_kets.start();

	 if(sym_formula_ == KUBO_BASTIN && double_buffer)	   
           filtered_polynomial_cycle_direct_doubleBuffer(bras, d_bras, rand_vec, s, 0);     
	 else
  	   filtered_polynomial_cycle_direct(bras, rand_vec, s, 0);
           //filtered_polynomial_cycle_OTF(bras, rand_vec, device_.damp_op(), device_.dis(), s, 0);
	 
	 csrmv_time_kets.stop("           Kets cycle time:            ");
         total_csrmv_time += csrmv_time_kets;


	 

	 
	 time_station_2 csrmv_time_bras;
         csrmv_time_bras.start();

	 if(sym_formula_ == KUBO_BASTIN && double_buffer)	   
	   filtered_polynomial_cycle_direct_doubleBuffer(kets, d_kets, rand_vec, s, 1);
	 else
	   filtered_polynomial_cycle_direct(kets, rand_vec, s, 1);
	 //filtered_polynomial_cycle_OTF(kets, rand_vec, device_.damp_op(), device_.dis(), s, 1);
	 
	 csrmv_time_bras.stop("           Bras cycle time:            ");
         total_csrmv_time += csrmv_time_bras;
	 

	 
	

	 time_station_2 FFTs_time;
	 FFTs_time.start();
	
	 if(sym_formula_ == KUBO_GREENWOOD)
	   Greenwood_FFTs(bras, kets, r_data, s);

	 else if(sym_formula_ == KUBO_BASTIN){
           if(double_buffer)
	     Bastin_FFTs_doubleBuffer(E_points, bras, d_bras, kets, d_kets, r_data, 1);
	   else{
	     std::cout<<"Yes. It is Bastin"<<std::endl;
	     Bastin_FFTs(E_points, bras,  kets, r_data, 1);
	   }
	 }
	 else if(sym_formula_ == KUBO_SEA){

	   std::cout<<"No. It is SEA  "<<sym_formula_<<std::endl;
	   Sea_FFTs(E_points, bras,  kets, r_data, 1);
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

       
      if(sym_formula_ == KUBO_BASTIN || sym_formula_ == KUBO_SEA )
        update_data_Bastin(E_points, r_data, final_data, conv_R, ( d - 1 ) * R + r, run_dir, filename);

	 
      plot_data(run_dir,filename);

      time_postProcess.stop( "       Post-processing time:       ");



      
      
      randVec_time.stop("       Total RandVec time:         ");
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

  if(sym_formula_ == KUBO_BASTIN && double_buffer){
    for(int m=0;m<M_dec;m++){
      delete []d_bras[m];
      delete []d_kets[m];
    }
  
    delete []d_bras;
    delete []d_kets;
  }  


  delete []rand_vec;
  delete []dmp_op;
  delete []dis_vec;
  
  /*-----------------------------------------------*/
  
  solver_station.stop("Total case execution time:              ");
}
