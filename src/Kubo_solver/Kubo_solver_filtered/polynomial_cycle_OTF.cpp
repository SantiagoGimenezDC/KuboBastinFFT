#include "../../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"


void Kubo_solver_filtered::filtered_polynomial_cycle_OTF(type** poly_buffer, type rand_vec[], r_type damp_op[], r_type dis_vec[], int s, int vel_op){
  
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


  
  type disp_factor = std::polar(1.0,   M_PI * ( - 2 * k_dis + initial_disp_ )  / M_ext );
  
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

