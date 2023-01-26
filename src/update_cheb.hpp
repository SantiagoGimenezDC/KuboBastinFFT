#ifndef UPDATE_CHEB_HPP
#define UPDATE_CHEB_HPP

#include "static_vars.hpp"
#include<complex>


const type t_standard_=2.7;


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
  
void update_cheb ( int m, type pre_factors[M_*M_], type polys[M_*SUBDIM_], type vec[DIM_], type p_vec[DIM_], type pp_vec[DIM_], type damp_op[DIM_], type a, type b){
  
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


     
      if(n>=C_*W_ && n<DIM_-C_*W_)
        for(int e=0;e<M_;e++)
	  polys[(n-C_*W_)*M_+e] += pre_factors[e*M_+m] * vec[n];

      
    }
 } 


 for(int n=0;n<DIM_;n++){
   pp_vec[n] = p_vec[n];
   p_vec[n]  = vec[n];
 }
 
};


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




//For tests alone

void update_cheb ( type vec[DIM_], type p_vec[DIM_], type pp_vec[DIM_], type a, type b){
  
  type t   = 2.0 * t_standard_/a,
       b_a = 2.0 * b/a;
  
  const int fullLe = 2*C_+LE_;

 
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W_; i++){
      int n = j * W_ + i;
      
      vec[n] = b_a * p_vec[n] - pp_vec[n];

      if( i!=0 )
	vec[n] += t * p_vec[n-1];
      
      if( i != (W_-1) )
	vec[n] += t * p_vec[n+1];
      
      if( j != (fullLe-1) && (j+i)%2!=0 )
	vec[n] += t * p_vec[n+W_];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] += t * p_vec[n-W_];

      
    }
 } 
 
};


/*
typedef std::complex<type> c_type;

void update_cheb_Rashba (int n, c_type pre_factors[E*M], c_type polys[E*subDim], type coup_str,type m_str, type a, type b, c_type ket[Dim], c_type p_ket[Dim], c_type pp_ket[Dim],  type disorder_potential[Dim],  type sFilter[Dim]){

  type t = t_standard_/a,
       b_a = b/a;
  
  m_str*=t;
  
 c_type f_y = 2.0*coup_str*t*cos(M_PI/6.0)/3.0,
        f_x = 2.0*coup_str*t/3.0,
        f_x2 = 2.0*coup_str*t*sin(M_PI/6.0)/3.0;


  f_y*=ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<M; i++){
      int n = j * M + i;

      ket(2*n)   =  2 * ( (b_a - m_str ) * p_ket(2*n)) - sFilter(2*n) * pp_ket(2*n);
      ket(2*n+1) =  2 * ( (b_a + m_str ) * p_ket(2*n+1)) - sFilter(2*n+1) * pp_ket(2*n+1);

      
      if( i!=0 ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n-2) + ( f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n-1) );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n-1) + ( -f_x2 * (((j+i)%2)==0? 1:-1)  + f_y ) * p_ket(2*n-2)  );
      }
      if(i != (M-1) ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n+2) + ( f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n+3)   );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n+3) + ( -f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n+2)   );
      }
      if(j != (fullLe-1)){
	ket(2*n)   += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M)    +  f_x * p_ket(2*n+2*M+1)) ;
	ket(2*n+1) += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M+1)  -  f_x * p_ket(2*n+2*M));
      }
      if(j != 0){
	ket(2*n)   += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M)   -  f_x * p_ket(2*n-2*M+1) );
	ket(2*n+1) += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M+1) +  f_x * p_ket(2*n-2*M)   );
      }
      
      ket(2*n)   *= sFilter(2*n);
      ket(2*n+1) *= sFilter(2*n+1);	       
    }
 }


 if(disorder_potential.size()!=0){
   int current=0;
   for(int c=0; c<nonContacts.rows(); c++){
    #pragma omp parallel for 
     for(int n=nonContacts(c,0); n<nonContacts(c,1); n++){
       ket(n)   +=  2 * sFilter(n) * disorder_potential(n-nonContacts(c,0)+current) * p_ket(n) / a;
     }
       current+=nonContacts(c,2);
   }
 }


 
 bool vertical_cbc=false;
 bool horizontal_cbc=false;


  if(M%2!=0&&vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;

  
 if(vertical_cbc)
  for(int j=0; j<fullLe; j++){

    int n_up = j * M +M-1;
    int n_down = j * M;


    ket(2*n_up)   += 2 * sFilter(2*n_up) * ( -t * p_ket(2*n_down) + ( -f_x2 * (((j)%2)==0? 1:-1) - f_y ) * p_ket(2*n_down+1) );
    ket(2*n_up+1) += 2 * sFilter(2*n_up+1) * ( -t * p_ket(2*n_down+1) + ( f_x2 * (((j)%2)==0? 1:-1)  - f_y ) * p_ket(2*n_down)  );
      
      
    ket(2*n_down)   += 2 * sFilter(2*n_down) * ( -t * p_ket(2*n_up) + ( -f_x2 * (((j+M-1)%2)==0? 1:-1) + f_y ) * p_ket(2*n_up+1)   );
    ket(2*n_down+1) += 2 * sFilter(2*n_down+1) * ( -t * p_ket(2*n_up+1) + ( f_x2 * (((j+M-1)%2)==0? 1:-1) + f_y ) * p_ket(2*n_up)   );
      
 } 


 if(fullLe%2!=0&&horizontal_cbc)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 
 if(horizontal_cbc)
    for(int i=0; i<M; i++){
      int n_front = i;
      int n_back = (fullLe-1) * M + i;

      
	ket(2*n_front)   += 2 * sFilter(2*n_front) * ((n_front+1)%2) * ( -t * p_ket(2*n_back) + ( -f_x * (((n_front)%2)==0? 1:-1) ) * p_ket(2*n_back+1) );
	ket(2*n_front+1) += 2 * sFilter(2*n_front+1) * ((n_front+1)%2) *( -t * p_ket(2*n_back+1) + ( f_x * (((n_front)%2)==0? 1:-1)  ) * p_ket(2*n_back)  );


	ket(2*n_back)   += 2 * sFilter(2*n_back) * ((n_front+1)%2) * ( -t * p_ket(2*n_front) + ( f_x * (((n_back)%2)==0? 1:-1)  ) * p_ket(2*n_front+1)   );
	ket(2*n_back+1) += 2 * sFilter(2*n_back+1) * ((n_front+1)%2) * ( -t * p_ket(2*n_front+1) + ( -f_x * (((n_back)%2)==0? 1:-1)  ) * p_ket(2*n_front)   );

    }


 
}; 




template<typename dataType>
void ArmchairGraphene<dataType>::vx_OTF (precision coup_str,precision m_str, precision a, precision b, VectorXpc& ket, VectorXpc& p_ket){

  int Le = this->parameters().Le_,
      M      = this->parameters().M_;
  
  precision t = t_standard_/a;

  if(this->parameters().C_%2==1)
    std::cout<<"BEWARE VX OP ONLY WORKS FOR EVEN C!!!!!!!!!!"<<std::endl;


      
  std::complex<precision> d_x=a0_,
                          d_x2=a0_*sin(M_PI/6.0),
    f_y = 2.0*coup_str*t*cos(M_PI/6.0)/3.0,
    f_x = 2.0*coup_str*t/3.0,
    f_x2 = 2.0*coup_str*t*sin(M_PI/6.0)/3.0;


//  

      
      if( i!=0 ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n-2) + ( f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n-1) );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n-1) + ( -f_x2 * (((j+i)%2)==0? 1:-1)  - f_y ) * p_ket(2*n-2)  );
      }
      if(i != (M-1) ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n+2) + ( f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n+3)   );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n+3) + ( -f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n+2)   );
      }
      if(j != (fullLe-1)){
	ket(2*n)   += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M)    +  f_x * p_ket(2*n+2*M+1)) ;
	ket(2*n+1) += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M+1)  -  f_x * p_ket(2*n+2*M));
      }
      if(j != 0){
	ket(2*n)   += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M)   -  f_x * p_ket(2*n-2*M+1) );
	ket(2*n+1) += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M+1) +  f_x * p_ket(2*n-2*M)   );
      }
      

///

  //  f_x*=ImUnit;
  // f_x2*=ImUnit;

  f_y*=ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<M; i++){
      int n = j * M + i;


        ket(2*n)   = 0;
	ket(2*n+1) = 0;

	
      if( i!=0 ){
	ket(2*n)   += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n-2) + ( f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n-1) );
	ket(2*n+1) += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n-1) + ( -f_x2 * (((j+i)%2)==0? 1:-1)  + f_y ) * p_ket(2*n-2)  );
      }
      if(i != (M-1) ){
	ket(2*n)   += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n+2) + ( f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n+3)   );
	ket(2*n+1) += d_x2 * (((j+i)%2)==0? -1:1) * ( -t * p_ket(2*n+3) + ( -f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n+2)   );
      }
      if(j != (Le-1)){
	ket(2*n)   +=   ((j+i)%2) * d_x * ( t * p_ket(2*n+2*M)    -  f_x * p_ket(2*n+2*M+1)) ;
	ket(2*n+1) +=   ((j+i)%2) * d_x * ( t * p_ket(2*n+2*M+1)  +  f_x * p_ket(2*n+2*M));
      }
      if(j != 0){
	ket(2*n)   +=  - ((j+i+1)%2) * d_x * (  t * p_ket(2*n-2*M)   +  f_x * p_ket(2*n-2*M+1));
	ket(2*n+1) +=  - ((j+i+1)%2) * d_x * (  t * p_ket(2*n-2*M+1) -  f_x * p_ket(2*n-2*M));
      }
      

    }
 } 



 bool vertical_cbc=false;
 bool horizontal_cbc=false;

 if(M%2!=0&&vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;
 
 if(vertical_cbc)
  for(int j=0; j<Le; j++){

    int n_up = j * M +M-1;
    int n_down = j * M;


    ket(2*n_up)   +=   d_x2 * (((j)%2)==0? 1:-1) * ( -t * p_ket(2*n_down) + ( -f_x2 * (((j)%2)==0? 1:-1) - f_y ) * p_ket(2*n_down+1) );
    ket(2*n_up+1) +=   d_x2 * (((j)%2)==0? 1:-1) * ( -t * p_ket(2*n_down+1) + ( f_x2 * (((j)%2)==0? 1:-1)  - f_y ) * p_ket(2*n_down)  );
      
      
    ket(2*n_down)   +=  -d_x2 * (((j+M)%2)==0? 1:-1) * ( -t * p_ket(2*n_up) + ( -f_x2 * (((j+M-1)%2)==0? 1:-1) + f_y ) * p_ket(2*n_up+1)   );
    ket(2*n_down+1) +=  -d_x2 * (((j+M)%2)==0? 1:-1) * ( -t * p_ket(2*n_up+1) + ( f_x2 * (((j+M-1)%2)==0? 1:-1) + f_y ) * p_ket(2*n_up)   );
      
 } 


 if(Le%2!=0&&horizontal_cbc)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;
 
 if(horizontal_cbc)
    for(int i=0; i<M; i++){
      int n_front = i;
      int n_back = (Le-1) * M + i;

      
        ket(2*n_front)   +=  - d_x * ((n_front+1)%2) * ( -t * p_ket(2*n_back) + ( f_x * (((n_front)%2)==0? 1:-1)  ) * p_ket(2*n_back+1) );
	ket(2*n_front+1) +=  - d_x * ((n_front+1)%2) * ( -t * p_ket(2*n_back+1) + ( -f_x * (((n_front)%2)==0? 1:-1)  ) * p_ket(2*n_back)  );


	ket(2*n_back)   +=  d_x * ((n_front)%2) * ( -t * p_ket(2*n_front) + ( -f_x * (((n_back)%2)==0? 1:-1) ) * p_ket(2*n_front+1)   );
	ket(2*n_back+1) +=  d_x * ((n_front)%2) * ( -t * p_ket(2*n_front+1) + ( f_x * (((n_back)%2)==0? 1:-1)  ) * p_ket(2*n_front)   );



    }


  std::complex<precision> im(0,1);
  ket*=-im;
 
}; 
*/

#endif //UPDATE_CHEB_HPP
