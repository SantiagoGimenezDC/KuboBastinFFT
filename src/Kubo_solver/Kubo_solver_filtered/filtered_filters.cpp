#include "../../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"

#include "../time_station.hpp"


void Kubo_solver_filtered::filter( int m, type* new_vec, type** poly_buffer, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
  int M         = parameters_.M_,
      M_ext     = filter_.parameters().M_ext_,
      num_parts = parameters_.num_parts_,
      SEC_SIZE  = parameters_.SECTION_SIZE_ ,
      size = SEC_SIZE;

  
  bool cyclic = true;
  
  int k_dis     = filter_.parameters().k_dis_,
      L         = filter_.parameters().L_,
      Np        = (L-1)/2,
      decRate   = filter_.parameters().decRate_;

  
  r_type KB_window[L];
 
  for(int i=0; i < L; i++)
    KB_window[i] = filter_.KB_window()[i];

  if( s != parameters_.num_parts_ - 1 )
    size -= device_.parameters().SUBDIM_ % parameters_.num_parts_;

  
  
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
      plus_eq( poly_buffer[ ( M - 1 + ( i + 1 ) ) / decRate ], tmp,  factor * KB_window[ Np + dist ], size );
	
	
    if( cyclic && i > M - 1 && ( ( i - 1 ) - ( M - 1 )  ) % decRate == 0 )
      plus_eq( poly_buffer[ ( i - M + 1 ) / decRate ], tmp,  factor * KB_window[ Np + dist  ], size );
  }
  
};



void Kubo_solver_filtered::filter_2( int m, type* new_vec, type** poly_buffer, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
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
    
    if( dist < Np || ( Np == 0 && dist == 0 ) )
      plus_eq( poly_buffer[ i ], tmp,  factor * KB_window[ Np + dist  ], size );
  }
  
};



void Kubo_solver_filtered::filter_2_doubleBuffer( int m, type* new_vec, type** poly_buffer, type** d_poly_buffer, type* tmp, type* tmp_velOp, int s, int vel_op ){

  
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

    if( dist < Np || ( Np == 0 && dist == 0 ) ){
      plus_eq( poly_buffer[ i ], tmp,  factor * KB_window[ Np + dist  ], size );
      plus_eq( d_poly_buffer[ i ], tmp,  m * factor * KB_window[ Np + dist  ], size );
    }
  }
};
