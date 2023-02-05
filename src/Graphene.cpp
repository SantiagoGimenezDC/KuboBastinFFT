#include "static_vars.hpp"
#include<iostream>
#include<complex>

#include"Graphene.hpp"

const type t_standard_=2.7;

Graphene::Graphene(device_vars& parameters) : graphene_vars_(parameters){

  if(this->parameters().C_==0)
    CYCLIC_BCs_=true;
  
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

 if(CYCLIC_BCs_){
   vertical_BC(vec,p_vec,damp_op,a);
   horizontal_BC(vec,p_vec,damp_op,a);  
 }
 
#pragma omp parallel for 
 for(int n=0;n<DIM;n++)
   p_vec[n]  = vec[n];
 

 
}


void Graphene::vertical_BC(type vec[], type p_vec[], type damp_op[], type a){
  type t   = 2.0 * t_standard_/a;

  int W = this->parameters().W_,
      LE = this->parameters().LE_,
      C = this->parameters().C_;


  const int fullLe = 2*C+LE;

  
  if(W%2!=0)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN W!!!!"<<std::endl;

  
#pragma omp parallel for 
  for(int j=0; j<fullLe; j++){
    int n_up = j * W +W-1;
    int n_down = j * W;

    vec[n_up]     += damp_op[n_up]   * t * p_vec[n_down];      
    vec[n_down]   += damp_op[n_down] * t * p_vec[n_up];      
 } 
}



void Graphene::horizontal_BC(type vec[], type p_vec[], type damp_op[], type a){
  type t   = 2.0 * t_standard_/a;

  int W = this->parameters().W_,
      LE = this->parameters().LE_,
      C = this->parameters().C_;


  const int fullLe = 2*C+LE;

  
  if(fullLe%2!=0)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 
#pragma omp parallel for 
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (fullLe-1) * W + i;
      vec[n_front]   +=  damp_op[n_front] * ( (n_front+1)%2 ) * t * p_vec[n_back];
      vec[n_back]    +=  damp_op[n_back]  * ( (n_front+1)%2 ) * t * p_vec[n_front];
    }

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




