#include "static_vars.hpp"
#include<complex>

#include"update_cheb.hpp"

const type t_standard_=2.7;

void update_cheb ( type vec[DIM_], type p_vec[DIM_], type pp_vec[DIM_], type damp_op[DIM_], type a, type b){

  type t   = 2.0 * t_standard_/a,
       b_a = 2.0 * b/a;
  
  const int fullLe = 2*C_+LE_;
     

  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W_; i++){
      int n = j * W_ + i;
      
      vec[n] = b_a * p_vec[n] - damp_op[n] * pp_vec[n];

      if( i!=0 )
	vec[n] += t * p_vec[n-1];
      
      if( i != (W_-1) )
	vec[n] += t * p_vec[n+1];
      
      if( j != (fullLe-1) && (j+i)%2!=0 )
	vec[n] += t * p_vec[n+W_];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] += t * p_vec[n-W_];

      
      vec[n] *= damp_op[n];
      
      pp_vec[n] = p_vec[n];
      
    }
 } 

#pragma omp parallel for 
 for(int n=0;n<DIM_;n++)
   p_vec[n]  = vec[n];
 

 
}




void update_cheb ( int m,  type polys[M_*SUBDIM_], type vec[DIM_], type p_vec[DIM_], type pp_vec[DIM_], type damp_op[DIM_], type a, type b){

  type t   = 2.0 * t_standard_/a,
       b_a = 2.0 * b/a;
  
  const int fullLe = 2*C_+LE_;
     

  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W_; i++){
      int n = j * W_ + i;
      
      vec[n] = b_a * p_vec[n] - damp_op[n] * pp_vec[n];

      if( i!=0 )
	vec[n] += t * p_vec[n-1];
      
      if( i != (W_-1) )
	vec[n] += t * p_vec[n+1];
      
      if( j != (fullLe-1) && (j+i)%2!=0 )
	vec[n] += t * p_vec[n+W_];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] += t * p_vec[n-W_];

      
      vec[n] *= damp_op[n];


     
      if(n>=C_*W_ && n<SUBDIM_+C_*W_)
	polys[(n-C_*W_)*M_+m] =  vec[n];

      
    }
 } 

#pragma omp parallel for 
 for(int n=0;n<DIM_;n++){
   pp_vec[n] = p_vec[n];
   p_vec[n]  = vec[n];
 } 
}
  


void vel_op (type vec[SUBDIM_], type p_vec[SUBDIM_] ){
  
  type dx1 = 1.0,
       dx2 = sin(M_PI/6.0),
       tx1 = dx1 * t_standard_,
       tx2 = dx2 * t_standard_;
  
 
#pragma omp parallel for 
 for(int j=0; j<LE_; j++){
    for(int i=0; i<W_; i++){
      int n = j * W_ + i;
      
      vec[n] = 0;

      if( i!=0 )
	vec[n] += tx2 * (((j+i)%2)==0? -1:1) * p_vec[n-1];
      
      if( i != (W_-1) )
	vec[n] += tx2 * (((j+i)%2)==0? -1:1) * p_vec[n+1];
      
      if( j != (LE_-1) && (j+i)%2!=0 )
	vec[n] += - tx1 * p_vec[n+W_];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] +=  tx1 * p_vec[n-W_];

      		     
    }
 } 

};




void batch_vel_op (type polys[M_*SUBDIM_], type tmp[SUBDIM_]){

  type  dx1 = 1.0,
        dx2 = sin(M_PI/6.0),
        tx1 = dx1 * t_standard_,
        tx2 = dx2 * t_standard_;

  
  for(int e=0; e<M_;e++){  
#pragma omp parallel for 
    for(int j=0; j<LE_; j++)
      for(int i=0; i<W_; i++){

	int n = j * W_ + i;
          tmp[n] = polys[n*M_+e];

      }
    
#pragma omp parallel for 
    for(int j=0; j<LE_; j++){
      for(int i=0; i<W_; i++){
        int n = j * W_ + i;


          polys[n*M_+e] = 0;
	  
          if( i !=0 )
	    polys[n*M_+e] += tx2 * (((j+i)%2)==0? -1:1) * tmp[n-1];
      
          if( i != (W_-1) )
	    polys[n*M_+e] += tx2 * (((j+i)%2)==0? -1:1) * tmp[n+1];
      
          if( j != (LE_-1) && (j+i)%2!=0 )
	    polys[n*M_+e] += - tx1 * tmp[n+W_];
      
          if( j != 0 && (j+i)%2==0 )
	    polys[n*M_+e] += tx1 * tmp[n-W_];
	        
      }      		     
    }
  } 

};



