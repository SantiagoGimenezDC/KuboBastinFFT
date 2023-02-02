#include "static_vars.hpp"
#include<complex>

#include"Graphene.hpp"

const type t_standard_=2.7;

Graphene::Graphene(device_vars& parameters) : graphene_vars_(parameters){

}


void Graphene::update_cheb ( type vec[], type p_vec[], type pp_vec[], type damp_op[], type a, type b){

  type t   = 2.0 * t_standard_/a,
       b_a = 2.0 * b/a;

  int W = this->parameters().W_,
      LE = this->parameters().LE_,
    C = this->parameters().C_,
    DIM = this->parameters().DIM_;


  const int fullLe = 2*C+LE;
     

  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;
      
      vec[n] = b_a * p_vec[n] - damp_op[n] * pp_vec[n];

      if( i!=0 )
	vec[n] += t * p_vec[n-1];
      
      if( i != (W-1) )
	vec[n] += t * p_vec[n+1];
      
      if( j != (fullLe-1) && (j+i)%2!=0 )
	vec[n] += t * p_vec[n+W];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] += t * p_vec[n-W];

      
      vec[n] *= damp_op[n];
      
      pp_vec[n] = p_vec[n];
      
    }
 } 
 
#pragma omp parallel for 
 for(int n=0;n<DIM;n++)
   p_vec[n]  = vec[n];
 

 
}




void Graphene::update_cheb ( int m, int M,  type polys[], type vec[], type p_vec[], type pp_vec[], type damp_op[], type a, type b){

  type t   = 2.0 * t_standard_/a,
       b_a = 2.0 * b/a;


  int W   = this->parameters().W_,
      LE  = this->parameters().LE_,
      C   = this->parameters().C_,
      DIM = this->parameters().DIM_,
      SUBDIM = this->parameters().SUBDIM_;    

  const int fullLe = 2*C+LE;
     

  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;
      
      vec[n] = b_a * p_vec[n] - damp_op[n] * pp_vec[n];

      if( i!=0 )
	vec[n] += t * p_vec[n-1];
      
      if( i != (W-1) )
	vec[n] += t * p_vec[n+1];
      
      if( j != (fullLe-1) && (j+i)%2!=0 )
	vec[n] += t * p_vec[n+W];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] += t * p_vec[n-W];

      
      vec[n] *= damp_op[n];


     
      if(n>=C*W && n<SUBDIM+C*W)
	polys[(n-C*W)*M+m] =  vec[n];

      
    }
 } 

#pragma omp parallel for 
 for(int n=0;n<DIM;n++){
   pp_vec[n] = p_vec[n];
   p_vec[n]  = vec[n];
 } 
}
  


void Graphene::vel_op (type vec[], type p_vec[] ){
  
  int W   = this->parameters().W_,
      LE  = this->parameters().LE_;    
  
  type dx1 = 1.0,
       dx2 = sin(M_PI/6.0),
       tx1 = dx1 * t_standard_,
       tx2 = dx2 * t_standard_;
  
 
#pragma omp parallel for 
 for(int j=0; j<LE; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;
      
      vec[n] = 0;

      if( i!=0 )
	vec[n] += tx2 * (((j+i)%2)==0? -1:1) * p_vec[n-1];
      
      if( i != (W-1) )
	vec[n] += tx2 * (((j+i)%2)==0? -1:1) * p_vec[n+1];
      
      if( j != (LE-1) && (j+i)%2!=0 )
	vec[n] += - tx1 * p_vec[n+W];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] +=  tx1 * p_vec[n-W];

      		     
    }
 } 

};




void Graphene::batch_vel_op (int M, type polys[], type tmp[]){


  int W   = this->parameters().W_,
      LE  = this->parameters().LE_;    

  
  type  dx1 = 1.0,
        dx2 = sin(M_PI/6.0),
        tx1 = dx1 * t_standard_,
        tx2 = dx2 * t_standard_;

  
  for(int e=0; e<M;e++){  
#pragma omp parallel for 
    for(int j=0; j<LE; j++)
      for(int i=0; i<W; i++){

	int n = j * W + i;
          tmp[n] = polys[n*M+e];

      }
    
#pragma omp parallel for 
    for(int j=0; j<LE; j++){
      for(int i=0; i<W; i++){
        int n = j * W + i;


          polys[n*M+e] = 0;
	  
          if( i !=0 )
	    polys[n*M+e] += tx2 * (((j+i)%2)==0? -1:1) * tmp[n-1];
      
          if( i != (W-1) )
	    polys[n*M+e] += tx2 * (((j+i)%2)==0? -1:1) * tmp[n+1];
      
          if( j != (LE-1) && (j+i)%2!=0 )
	    polys[n*M+e] += - tx1 * tmp[n+W];
      
          if( j != 0 && (j+i)%2==0 )
	    polys[n*M+e] += tx1 * tmp[n-W];
	        
      }      		     
    }
  } 

};



