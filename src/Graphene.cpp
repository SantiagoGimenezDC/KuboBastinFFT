#include "static_vars.hpp"
#include<iostream>
#include<complex>
#include<random>
#include<chrono>
#include<eigen-3.4.0/Eigen/Core>


#include"Graphene.hpp"



Graphene::Graphene(device_vars& parameters) : graphene_vars_(parameters), rng_(parameters.dis_seed_){

  if(this->parameters().C_==0)
    CYCLIC_BCs_=true;
  
}

void Graphene::Anderson_disorder(r_type disorder_vec[]){

  int SUBDIM = graphene_vars_.SUBDIM_;
  r_type str = graphene_vars_.dis_str_;
  
  for(int i=0;i<SUBDIM; i++){
    r_type random_potential = str * this->rng().get()-str/2;

    disorder_vec[i] = random_potential;
  }
  
}


void Graphene::update_cheb ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[], r_type a, r_type b){

  r_type t   = 2.0 * t_standard_/a,
       b_a = 2.0 * b/a;

  int W = this->parameters().W_,
    LE  = this->parameters().LE_,
    C   = this->parameters().C_,
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
 for(int n=0; n<LE*W; n++)
   vec[C*W+n] += dis_vec[n] * p_vec[C*W+n] / a;

 
 if(CYCLIC_BCs_){
   vertical_BC(vec,p_vec,damp_op,a);
   horizontal_BC(vec,p_vec,damp_op,a);  
 }
 
#pragma omp parallel for 
 for(int n=0;n<DIM;n++)
   p_vec[n]  = vec[n];
 
}


void Graphene::update_cheb ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type a, r_type b){

  r_type t   = 2.0 * t_standard_/a,
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


void Graphene::vertical_BC(type vec[], type p_vec[], r_type damp_op[], r_type a){
  r_type t   = 2.0 * t_standard_/a;

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



void Graphene::horizontal_BC(type vec[], type p_vec[], r_type damp_op[], r_type a){
  r_type t   = 2.0 * t_standard_/a;

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



void Graphene::update_cheb ( int m, int M,  type polys[], type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type a, r_type b){

  r_type t   = 2.0 * t_standard_/a,
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



void Graphene::H_ket ( type vec[], type p_vec[], r_type a, r_type b){

  r_type t = t_standard_/a,
       b_a = b/a;


  int W   = this->parameters().W_,
      LE  = this->parameters().LE_,
    C   = this->parameters().C_;
  
  const int fullLe = 2*C+LE;
     
  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;
      
      vec[n] = b_a * p_vec[n];

      if( i!=0 )
	vec[n] += t * p_vec[n-1];
      
      if( i != (W-1) )
	vec[n] += t * p_vec[n+1];
      
      if( j != (fullLe-1) && (j+i)%2!=0 )
	vec[n] += t * p_vec[n+W];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] += t * p_vec[n-W];

      
    }
 } 
 
}



void Graphene::vel_op (type vec[], type p_vec[] ){
  
  int W   = this->parameters().W_,
      LE  = this->parameters().LE_;    
  
  r_type dx1 = 1.0,
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




void Graphene::minMax_EigenValues( int maxIter, r_type& eEmax, r_type& eEmin){ //Power Method; valid if eigenvalues are real
  int DIM = graphene_vars_.DIM_;

  Eigen::Matrix<type, -1, 1> y = Eigen::Matrix<type, -1, 1>::Constant(DIM,1, 1.0/sqrt(DIM) ),
            y_Ant=y;
 

  r_type y_norm = 0;
  r_type Emax, Emin;

  std::cout<<"   Calculating Energy band bounds:    "<<std::endl;
  auto start = std::chrono::steady_clock::now();

  
  for( int i=0; i<maxIter; i++){
    
    this->H_ket(y.data(),y_Ant.data(),1.0,0.0);
    y_norm=y.norm();

    y=y/y_norm;
    y_Ant=y;
  }

  
  this->H_ket(y.data(),y_Ant.data(),1.0,0.0);

  Emax = std::real(y_Ant.dot(y)/y_Ant.squaredNorm());



  
  y  =  Eigen::Matrix<type, -1, 1>::Constant(DIM,1, 1.0/sqrt(DIM) );
    
  for( int i=0; i<maxIter; i++){
    this->H_ket(y.data(),y_Ant.data(),1.0,-Emax);
    y_norm=y.norm();

    y=y/y_norm;
    y_Ant=y;  
  }


  this->H_ket(y.data(),y_Ant.data(),1.0,-Emax);
  
  Emin  = std::real(((y_Ant.dot(y))/y_Ant.squaredNorm()));
  Emin += Emax;

   
  auto end = std::chrono::steady_clock::now();
  std::cout<<"   Time to perform Lanczos Recursion:    ";
  int millisec0=std::chrono::duration_cast<std::chrono::milliseconds>
                (end - start).count();
  int sec0=millisec0/1000, min0=sec0/60, reSec0=sec0%60;
  std::cout<<min0<<" min, "<<reSec0<<" secs;"<<
	     " ("<< millisec0<<"ms) "<<std::endl<<std::endl;     


  
  eEmin  = std::min(Emax,Emin);
  eEmax  = std::max(Emax,Emin);
  
    
  std::cout<<"    Highest absolute energy bound:    "<<eEmax<<std::endl;
  std::cout<<"    Lower absolute energy bound:      "<<eEmin<<std::endl<<std::endl<<std::endl;
}


