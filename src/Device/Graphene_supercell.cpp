#include<iostream>
#include<fstream>
#include<complex>
#include<random>
#include<chrono>
#include<eigen-3.4.0/Eigen/Core>


#include"Graphene_supercell.hpp"


Graphene_supercell::Graphene_supercell(device_vars& parameters) : Graphene(parameters){
    int W     = parameters.W_,
        Le     = parameters.LE_,
        C      = parameters.C_,
        fullLe = (2*C+Le);

    fullLe_ = fullLe;

    if(W%2!=0)
      std::cout<<"please choose even W"<<std::endl;
    if(this->parameters().C_==0)
      CYCLIC_BCs_=true;

    //              CYCLIC_BCs_=false;

    this->set_sysLength( (fullLe-1) * (1.0+sin(M_PI/6)) ); 
    this->set_sysSubLength( (Le-1)*(1.0+sin(M_PI/6)) );

    //Bz here will be trated as the ratio between phi/phi_0;
    peierls_d_ = 2.0 * M_PI * this->parameters().Bz_ / double(2*(W-1));
  
    print_hamiltonian();

    this->set_coordinates();
    
}


void Graphene_supercell::print_hamiltonian(){
  int dim = this->parameters().DIM_;
  int DIM = this->parameters().DIM_;

  Eigen::MatrixXcd H_r(dim,dim), S(dim,dim);

  std::ofstream dataP;
  dataP.open("update_spc.txt");
        
    for(int j=0;j<dim;j++){
      for(int i=0;i<dim;i++){

	Eigen::Matrix<std::complex<double>, -1, 1>  term_i=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM), tmp=term_i, term_j=term_i, null=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM);
        Eigen::Matrix<double, -1, 1>  null2=Eigen::Matrix<double, -1, 1>::Zero(DIM);
	term_i(i)=1;
	term_j(j)=1;


        this->update_cheb(tmp.data(),term_j.data(),null.data());
	//this->H_ket(tmp.data(),term_j.data());
        //vel_op_x(tmp.data(),term_j.data());
        //vel_op_y(tmp.data(),term_j.data());
	std::complex<double> termy = term_i.dot(tmp);

	 H_r(i,j)=termy;
      }
    }

    dataP<<H_r;//.real();

    std::cout<<(H_r-H_r.adjoint()).norm()<<std::endl;
  dataP.close();

}



void Graphene_supercell::update_cheb ( type vec[], type p_vec[], type pp_vec[]){

  r_type t = 2.0 * t_a_,
       b_a = 2.0 * b_/a_;

  r_type *damp_op=this->damp_op(),
    *dis_vec = this->dis();

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
	vec[n] += t * p_vec[n-1] * peierls(i,-1);
      
      if( i != ( W - 1 ) )
	vec[n] += t * p_vec[n+1] * peierls(i,1);
      
      if( j != ( fullLe - 1 ) && !(j == 0 && i == W-1) && i%2 == 0 )
	vec[n] += t * p_vec[ n + W + 1];
      
      if( j != 0 && i%2 != 0 )
	vec[n] += t * p_vec[ n - W - 1];



      /*      
      if(!CYCLIC_BCs_ && C==0){      
        if(( i == (W-1) && j==0 ) || (i == 1 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n-1] * peierls(i,-1);

        if( ( i == (W-2) && j==0 ) || (i == 0 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n+1] * peierls(i,1);
      }
      */
      
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





void Graphene_supercell::vertical_BC(r_type a2, type vec[], type p_vec[], r_type damp_op[]){
  r_type t   = a2 * t_a_;

  int W = this->parameters().W_,
      LE = this->parameters().LE_,
      C = this->parameters().C_;


  const int fullLe = 2*C+LE;

  

  
#pragma omp parallel for 
  for(int j=0; j<fullLe; j++){
    int n_up = j * W +W-1;
    int n_down = j * W;


      vec[n_up]     += damp_op[n_up] * t * p_vec[n_down] * peierls(W-1,1);
      vec[n_down]   += damp_op[n_down] * t * p_vec[n_up] * peierls(0,-1);
    
  } 
}



void Graphene_supercell::horizontal_BC(r_type a2, type vec[], type p_vec[], r_type damp_op[]){
  r_type t   = a2 *  t_a_;

  int W = this->parameters().W_,
      LE = this->parameters().LE_,
      C = this->parameters().C_;

  
  const int fullLe = 2*C+LE;

  
  if(fullLe%2==0)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR ODD LE+C!!!!"<<std::endl;


#pragma omp parallel for 
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (fullLe-1) * W + i;
      vec[n_front]   +=  damp_op[n_front] * ( ( n_front )%2!=0 ) * t * p_vec[n_back-1];
      vec[n_back]    +=  damp_op[n_back]  * ( ( n_back )%2==0 ) * t * p_vec[n_front+1];
      
    }
}



void Graphene_supercell::H_ket ( type vec[], type p_vec[]){

  r_type t = t_a_,
       b_a = b_/a_;

  r_type *damp_op=this->damp_op(),
         *dis = this->dis();
  

  
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
	vec[n] += t * p_vec[n-1] * peierls(i,-1);
      
      if( i != ( W - 1 ) )
	vec[n] += t * p_vec[n+1] * peierls(i,1);
      
      if( j != ( fullLe - 1 ) && !(j == 0 && i == W-1) && i%2 == 0 )
	vec[n] += t * p_vec[ n + W + 1];
      
      if( j != 0 && i%2 != 0 )
	vec[n] += t * p_vec[ n - W - 1];



      
      /*
      if(!CYCLIC_BCs_ && C==0){            
        if( (i == (W-1) && j==0) || (i == 1 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n-1] * peierls(i,-1);

        if( (i == (W-2) && j==0) || (i == 0 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n+1] * peierls(i,1);
      }
      */
      
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



void Graphene_supercell::vel_op_x (type vec[], type p_vec[] ){
  
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
	  vec[n] += tx2 * ((i%2)==0? -1:1) * p_vec[n-1] * peierls(i,-1);
      
        if( i != (W-1) )
	  vec[n] += tx2 * ((i%2)==0? -1:1) * p_vec[n+1] * peierls(i,1);
      
        if( j != (LE-1) && !(j == C && i == W-1) && i%2==0 )
	  vec[n] += - tx1 * p_vec[n+W+1];
      
        if( j != 0 && i%2!=0 )
	  vec[n] +=  tx1 * p_vec[n-W-1];


	/*
      if(!CYCLIC_BCs_ && C==0){            
        if( (i == (W-1) && j==0 ) || (i == 1 && j==LE-1))
	  vec[n] -= tx2 * ((i%2)==0? -1:1) * p_vec[n-1] * peierls(i,-1);

        if( ( i == (W-2) && j==0) || (i == 0 && j==LE-1) )
	  vec[n] -= tx2 * ((i%2)==0? -1:1) * p_vec[n+1] * peierls(i,1);	  
      }      
	*/
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
      vec[n_front]   +=  - ( (n_front)%2!=0 ) * tx1 * p_vec[n_back-1];
      vec[n_back]    +=    ( (n_back)%2==0 ) * tx1 * p_vec[n_front+1];
    }
  }


  
};



void Graphene_supercell::vel_op_y (type vec[], type p_vec[] ){
  
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

	/*
      if(!CYCLIC_BCs_ && C==0){            
        if( (i == (W-1) && j==0 ) || (i == 1 && j==LE-1))
	  vec[n] -=  - ty2 * p_vec[n-1] * peierls(i,-1);

        if( ( i == (W-2) && j==0) || (i == 0 && j==LE-1) )
	  vec[n] -= ty2 * p_vec[n+1] * peierls(i,1);
      }      
	*/

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
