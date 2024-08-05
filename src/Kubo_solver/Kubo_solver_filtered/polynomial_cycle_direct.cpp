#include "../../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"



void Kubo_solver_filtered::filtered_polynomial_cycle_direct(type** poly_buffer, type rand_vec[],  int s, int vel_op){
  
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



void Kubo_solver_filtered::filtered_polynomial_cycle_direct_doubleBuffer(type** poly_buffer, type** d_poly_buffer, type rand_vec[],  int s, int vel_op){
  
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

  
 
  filter_2_doubleBuffer( 0, pp_vec, poly_buffer, d_poly_buffer, tmp, tmp_velOp, s, vel_op );  


  
  device_.H_ket ( p_vec, pp_vec );
  filter_2_doubleBuffer( 1, p_vec, poly_buffer, d_poly_buffer, tmp, tmp_velOp, s, vel_op ); 


  for( int m=2; m<M; m++ ){

    device_.update_cheb( vec, p_vec, pp_vec );

    filter_2_doubleBuffer( m, vec, poly_buffer, d_poly_buffer, tmp, tmp_velOp, s, vel_op );
  }
      
    delete []vec;
    delete []p_vec;
    delete []pp_vec;
    delete []tmp;
    delete []tmp_velOp;
  
}

