#include "../../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"



void Kubo_solver_filtered::filtered_polynomial_cycle_OTF_imag(type** re_poly_buffer, type** im_poly_buffer, type rand_vec[],  int s, int vel_op){
  
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

  //  r_type *damp_op, *dis_vec;
  
  type disp_factor = std::polar(1.0,   M_PI * ( - 2 * k_dis + initial_disp_ )  / M_ext );
  
  r_type KB_window[L];

  bool cyclic = true;

  /*
  if( M_ext > M + Np)
    cyclic = false;
  */
  
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
    device_.vel_op( pp_vec, rand_vec, parameters_.vel_dir_2_  );  
  else
#pragma omp parallel for
    for(int l = 0; l < DIM; l++)
      pp_vec[l] = - rand_vec[l]; //This minus sign is due to the CONJUGATION of applying both velocity operators to the KET side!!!!

  
  
  //=================================KPM Step 0======================================//


    filter_imag( 0, pp_vec, re_poly_buffer, im_poly_buffer, tmp, tmp_velOp, s, vel_op );  

  
  
  //=================================KPM Step 1======================================//     
    
  device_.H_ket ( p_vec, pp_vec);
  
    
  type factor = std::polar(1.0,  M_PI * 1 * (  - 2 * k_dis + initial_disp_ ) / M_ext );


  //Building first filtered recursion vector
  plus_eq(pp_vec_f, p_vec, 2 * factor * KB_window[0], DIM);

  filter_imag( 1, p_vec, re_poly_buffer, im_poly_buffer, tmp, tmp_velOp, s, vel_op ); 

  



  

  
  //=================================KPM Steps 2 and so on===============================//

    for( int m=2; m<M; m++ ){

      device_.update_cheb( vec, p_vec, pp_vec);

      

      if( m < L || M - m - 1 < L )
        filter_imag( m, vec, re_poly_buffer, im_poly_buffer, tmp, tmp_velOp, s, vel_op ); 
      
      
      
      //==========================Filtered recursion==============================//
      factor = 2 * std::polar(1.0,  M_PI * m * (  - 2 * k_dis + initial_disp_) / M_ext );

      
      if( m < L + 1 )
        plus_eq( pp_vec_f, vec,  factor * KB_window[ Np + ( m - ( Np + 1 ) ) ], DIM);

      if( m < L + 2  &&  m > 1)
        plus_eq( p_vec_f,  vec,  factor * KB_window[ Np + ( m - ( Np + 2 ) ) ], DIM);


      

      
                                   //EXCEPTIONAL CASES
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
      if( (Np + 1 ) % decRate == 0 && m == L + 2){
	  if( vel_op == 1 ){
            device_.vel_op( tmp_velOp, pp_vec_f, parameters_.vel_dir_2_ );
            device_.traceover( tmp, tmp_velOp, s, num_parts );
	    plus_eq_imag( re_poly_buffer[ ( Np + 1 ) / decRate ], im_poly_buffer[  ( Np + 1 ) / decRate ], tmp,  1.0, SEC_SIZE);
	  }
          else{	    
	    device_.traceover( tmp, pp_vec_f, s, num_parts );
	    plus_eq_imag( re_poly_buffer[ ( Np + 1 ) / decRate ], im_poly_buffer[  ( Np + 1 ) / decRate ], tmp,  1.0, SEC_SIZE);
	  }
	
	  if( (Np + 2) % decRate == 0 ){
	    if( vel_op == 1 ){
              device_.vel_op( tmp_velOp, p_vec_f, parameters_.vel_dir_2_ );
              device_.traceover( tmp, tmp_velOp, s, num_parts );
	      plus_eq_imag( re_poly_buffer[ ( Np + 2 ) / decRate ], im_poly_buffer[  ( Np + 2 ) / decRate ], tmp,  1.0 , SEC_SIZE);
	    }
            else{
	      device_.traceover( tmp, p_vec_f, s, num_parts );
	      plus_eq_imag( re_poly_buffer[ ( Np + 2 ) / decRate ], im_poly_buffer[  ( Np + 2 ) / decRate ], tmp,  1.0 , SEC_SIZE);
	    }
	  }
      }
      if( (Np + 2) % decRate == 0 && (Np + 1) % decRate > 0 && m == L + 2 ){
	  if( vel_op == 1 ){
            device_.vel_op( tmp_velOp, p_vec_f, parameters_.vel_dir_2_ );
	    plus_eq_imag( re_poly_buffer[ Np / decRate + 1 ], im_poly_buffer[ Np / decRate + 1 ], tmp,  1.0 , SEC_SIZE);
	    
	  }
          else{
	    device_.traceover( tmp, p_vec_f, s, num_parts );
	    plus_eq_imag( re_poly_buffer[ Np / decRate + 1 ], im_poly_buffer[ Np / decRate + 1 ], tmp,  1.0 , SEC_SIZE);
	  }
      }
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
      //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

      
      //      plus_eq_imag( poly_buffer_re[ i ], poly_buffer_im[ i ], tmp,  factor * KB_window[ Np + dist  ], size);
      

      //Filtered recursion starts at m = Np, ends at M - Np
      
      if( m == M-1)
      for(int i=0; i<M_dec;i++)
	std::cout<<i<<"/"<<M_dec<<"   "<<re_poly_buffer[i][SEC_SIZE/2]<<std::endl;
      
      
      if( m > L + 1 && m < M){


	device_.update_cheb_filtered ( vec_f, p_vec_f, pp_vec_f, NULL, NULL, disp_factor);

	
	if( ( m - Np ) % decRate == 0 ){
	  if( vel_op == 1 ){
            device_.vel_op( tmp_velOp, vec_f, parameters_.vel_dir_2_ );
            device_.traceover( tmp, tmp_velOp, s, num_parts );
	    plus_eq_imag( re_poly_buffer[ ( m - Np ) / decRate ], im_poly_buffer[  ( m - Np ) / decRate ], tmp,  1.0 , SEC_SIZE);

	  }
          else	{
	    device_.traceover( tmp, vec_f, s, num_parts );
	    plus_eq_imag( re_poly_buffer[ ( m - Np ) / decRate ], im_poly_buffer[ ( m - Np ) / decRate ], tmp,  1.0 , SEC_SIZE);
	  }
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
