#include<iostream>
#include<complex>
#include<random>
#include<chrono>
#include<eigen-3.4.0/Eigen/Core>


#include"Graphene.hpp"
#include<fstream>

void Graphene::print_hamiltonian(){
  int dim = 10;//this->parameters().DIM_;
  int DIM = this->parameters().DIM_;

  Eigen::MatrixXcd H_r(dim,dim), S(dim,dim);

  std::ofstream dataP;
  dataP.open("Hamiltonian.txt");
        

    for(int j=0;j<dim;j++){
      for(int i=0;i<dim;i++){

	Eigen::Matrix<std::complex<double>, -1, 1>  term_i=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM), tmp=term_i, term_j=term_i, null=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM);
        Eigen::Matrix<double, -1, 1>  null2=Eigen::Matrix<double, -1, 1>::Zero(DIM);
	term_i(i)=1;
	term_j(j)=1;


        //update_cheb(tmp.data(),term_j.data(),null.data());
        vel_op_otf(tmp.data(),term_j.data());

	std::complex<double> termy = term_i.dot(tmp);

	 H_r(i,j)=termy;
      }
    }

    dataP<<H_r;

  dataP.close();

}

Graphene::Graphene(device_vars& parameters) : Device(parameters),   coordinates_(parameters.W_,parameters.LE_,parameters.C_){
    int W     = this->parameters().W_,
        Le     = this->parameters().LE_,
        C      = this->parameters().C_,
        fullLe = (2*C+Le);

    fullLe_ = fullLe;
    
  if(this->parameters().C_==0)
    CYCLIC_BCs_=true;



  this->set_sysLength( (fullLe-1) * (1.0+sin(M_PI/6)) ); 
  this->set_sysSubLength( (Le-1)*(1.0+sin(M_PI/6)) );

  //Bz here will be trated as the ratio between phi/phi_0;
  peierls_d_ = 2.0 * M_PI * this->parameters().Bz_ / double(2*(W-1));
  
  //print_hamiltonian();

  this->set_coordinates();
  
}


void Graphene::rearrange_initial_vec(type r_vec[]){ //supe duper hacky
  int Dim = this->parameters().DIM_,
    subDim = this->parameters().SUBDIM_;

  int C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;

  type tmp[subDim];

#pragma omp parallel for
    for(int n=0;n<subDim;n++)
      tmp[n]=r_vec[n];

#pragma omp parallel for
    for(int n=0;n<Dim;n++)
      r_vec[n] = 0;
        

#pragma omp parallel for
    for(int n=0;n<Le*W;n++)
      r_vec[C*W + n ]=tmp[ n];

}


void Graphene::traceover(type* traced, type* full_vec, int s, int num_reps){
  int subDim = this->parameters().SUBDIM_,
      C   = this->parameters().C_,
      W   = this->parameters().W_,
      sec_size = subDim/num_reps,
      buffer_length = sec_size;
	
  if( s == num_reps-1 )
      buffer_length += subDim % num_reps;

      
#pragma omp parallel for 
      for(int i=0;i<buffer_length;i++)
        traced[i] = full_vec[s*sec_size + i+C*W];

  };


void Graphene::update_cheb_filtered ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[], type disp_factor){

  r_type t = 2.0 * t_a_,
       b_a = 2.0 * b_/a_;


  int W = this->parameters().W_,
    LE  = this->parameters().LE_,
    C   = this->parameters().C_,
    DIM = this->parameters().DIM_;


  const int fullLe = 2*C+LE;
     

  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;
      
      vec[n] = b_a * p_vec[n] - disp_factor * damp_op[n] * pp_vec[n];

      if( i!=0 )
	vec[n] += t * p_vec[n-1];
      
      if( i != (W-1) )
	vec[n] += t * p_vec[n+1];
      
      if( j != (fullLe-1) && (j+i)%2!=0 )
	vec[n] += t * p_vec[n+W];
      
      if( j != 0 && (j+i)%2==0 )
	vec[n] += t * p_vec[n-W];

      
      vec[n] *= ( disp_factor * damp_op[n]);
      
      pp_vec[n] = p_vec[n];
      
    }
 } 

#pragma omp parallel for 
 for(int n=0; n<LE*W; n++)
   vec[C*W+n] += disp_factor * dis_vec[n] * p_vec[C*W+n] / a_;

 
 if(CYCLIC_BCs_){
   vertical_BC(2.0, vec,p_vec,damp_op);
   horizontal_BC(2.0, vec,p_vec,damp_op);  
 }
 
#pragma omp parallel for 
 for(int n=0;n<DIM;n++)
   p_vec[n]  = vec[n];
 
}





void Graphene::update_cheb_otf ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[]){

  r_type t = 2.0 * t_a_,
       b_a = 2.0 * b_/a_;


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

      if( i != 0 )
	vec[n] += t * p_vec[n-1] * peierls(i,-1);// std::polar(1.0,  - ( i % 2 == 0 ? -1 : 1 ) * peierls_d_ * ( - 2 * i + 1  ) );
      
      if( i != ( W - 1 ) )
	vec[n] += t * p_vec[n+1] * peierls(i,1);// * std::polar(1.0,   ( i % 2 == 0 ? -1 : 1 ) * peierls_d_ * (  2 * i + 1  ) );
      
      if( j != ( fullLe - 1 ) && ( j + i ) % 2 != 0 )
	vec[n] += t * p_vec[ n + W ];
      
      if( j != 0 && ( j + i ) %2 == 0 )
	vec[n] += t * p_vec[ n - W ];

      
      vec[n] *= damp_op[n];
      
      pp_vec[n] = p_vec[n];
      
    }
 } 

#pragma omp parallel for 
 for(int n=0; n<LE*W; n++)
   vec[C*W+n] += dis_vec[n] * p_vec[C*W+n] / a_;

 
 if(CYCLIC_BCs_){
   vertical_BC(2.0, vec,p_vec,damp_op);
   horizontal_BC(2.0, vec,p_vec,damp_op);  
 }
 
#pragma omp parallel for 
 for(int n=0;n<DIM;n++)
   p_vec[n]  = vec[n];
 
}





void Graphene::vertical_BC(r_type a2, type vec[], type p_vec[], r_type damp_op[]){
  r_type t   = a2 * t_a_;

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

    vec[n_up]     += damp_op[n_up] * t * p_vec[n_down] * peierls(W-1,1); //* std::polar(1.0,   ( ( W - 1 ) % 2 == 0 ? -1 : 1 ) * peierls_d_ * ( - 2 * (W - 1) + 1  ) );
     
    vec[n_down]   += damp_op[n_down] * t * p_vec[n_up] * peierls(0,-1); //* std::polar(1.0,  - ( 0 % 2 == 0 ? -1 : 1 ) * peierls_d_ * (  2 * 0 + 1  ) );
            
  } 
}



void Graphene::horizontal_BC(r_type a2, type vec[], type p_vec[], r_type damp_op[]){
  r_type t   = a2 *  t_a_;

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
      vec[n_front]   +=  damp_op[n_front] * ( ( n_front )%2 ) * t * p_vec[n_back];
  
      vec[n_back]    +=  damp_op[n_back]  * ( ( n_back )%2 ) * t * p_vec[n_front];
    }
}



void Graphene::H_ket_otf ( type vec[], type p_vec[], r_type damp_op[], r_type dis[]){

  r_type t = t_a_,
       b_a = b_/a_;


  
  int W   = this->parameters().W_,
      LE  = this->parameters().LE_,
    C   = this->parameters().C_;
  
  const int fullLe = 2*C+LE;
     
  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;
      
      vec[n] = b_a * p_vec[n];

      if( i != 0 )
	vec[n] += t * p_vec[n-1] * peierls(i,-1);//* std::polar(1.0,  - ( i % 2 == 0 ? -1 : 1 ) * peierls_d_ * ( - 2 * i + 1  ) );
      
      if( i != ( W - 1 ) )
	vec[n] += t * p_vec[n+1] * peierls(i,1);
      
      if( j != ( fullLe - 1 ) && ( j + i ) % 2 != 0 )
	vec[n] += t * p_vec[ n + W ];
      
      if( j != 0 && ( j + i ) %2 == 0 )
	vec[n] += t * p_vec[ n - W ];

      
      vec[n] *= damp_op[n];      
    }
 } 

#pragma omp parallel for 
 for(int n=0; n<LE*W; n++)
   vec[C*W+n] += dis[n] * p_vec[C*W+n] / a_;



 if(CYCLIC_BCs_){
   vertical_BC(1.0, vec,p_vec,damp_op);
   horizontal_BC(1.0, vec,p_vec,damp_op);  
 }
}



void Graphene::vel_op_otf (type vec[], type p_vec[] ){
  
  int W   = this->parameters().W_,
      LE  = this->parameters().LE_,
      C  = this->parameters().C_;    
  
  r_type dx1 = 1.0,
         dx2 = sin(M_PI/6.0),
         tx1 = dx1 * t_standard_,
         tx2 = dx2 * t_standard_;


 
#pragma omp parallel for 
  for(int j=0; j< ( LE + 2 * C ); j++){
    for(int i=0; i<W; i++){
      int n =  j * W + i;
      
      vec[n] = 0;

      if( n >= C * W ){
        if( i!=0 )
	  vec[n] += tx2 * (((j+i)%2)==0? -1:1) * p_vec[n-1] * peierls(i,-1);
      
        if( i != (W-1) )
	  vec[n] += tx2 * (((j+i)%2)==0? -1:1) * p_vec[n+1] * peierls(i,1);
      
        if( j != (LE-1) && (j+i)%2!=0 )
	  vec[n] += - tx1 * p_vec[n+W];
      
        if( j != 0 && (j+i)%2==0 )
	  vec[n] +=  tx1 * p_vec[n-W];
      }
    }
  }

  

    
  if(CYCLIC_BCs_){
#pragma omp parallel for 
    for(int j=0; j< ( LE + 2 * C ); j++){
      int n_up = j * W + W-1;
      int n_down = j * W;

      vec[n_up]     +=  tx2 * (((j+W-1)%2)==0? -1:1) * p_vec[n_down] * peierls(W-1,1);
      vec[n_down]   +=  tx2 * (((j+W-1)%2)==0? -1:1) * p_vec[n_up]   * peierls(0,-1);
    }

 
#pragma omp parallel for 
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (LE + 2 * C - 1) * W + i;
      vec[n_front]   +=  - ( (n_front)%2 ) * tx1 * p_vec[n_back];
      vec[n_back]    +=    ( (n_front)%2 ) * tx1 * p_vec[n_front];
    }
  }


  
};



void Graphene::vel_op_y (type vec[], type p_vec[] ){
  
  int W   = this->parameters().W_,
      LE  = this->parameters().LE_,
      C  = this->parameters().C_;    
  
  r_type dy2 = cos(M_PI/6.0),
         ty2 = dy2 * t_standard_;
  
  std::complex<r_type> Im(0,1.0);


#pragma omp parallel for 
  for(int j=0; j<(LE+2*C); j++){
    for(int i=0; i<W; i++){
      int n =  j * W + i;
      
      vec[n] = 0;

      if( n >= C * W ){
        if( i!=0 )
	  vec[n] +=  - ty2 * p_vec[n-1] * peierls(i,-1);
      
        if( i != (W-1) )
	  vec[n] += ty2 * p_vec[n+1] * peierls(i,1);
      }
    }
  } 


  if(CYCLIC_BCs_){
#pragma omp parallel for 
    for(int j=0; j< ( LE + 2 * C ); j++){
      int n_up = j * W + W-1;
      int n_down = j * W;

      vec[n_up]     +=  - ty2 * p_vec[n_down]  * peierls(W-1,1);
      vec[n_down]   +=  + ty2 * p_vec[n_up]    * peierls(0,-1);
    }

  }

};





