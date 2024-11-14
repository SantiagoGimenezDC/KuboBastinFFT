#include<iostream>
#include<fstream>
#include<complex>
#include<random>
#include<chrono>
#include<eigen3/Eigen/Core>


#include"Graphene_supercell.hpp"


Graphene_supercell::Graphene_supercell(device_vars& parameters) : Graphene(parameters){
    int W     = parameters.W_,
        Le     = parameters.LE_,
        C      = parameters.C_,
        fullLe = (2*C+Le);

    fullLe_ = fullLe;

    if(this->parameters().W_%2!=0 )
      std::cout<<"Graphene supercell only valid for EVEN W !!"<<std::endl;


    if(this->parameters().C_==0)
      CYCLIC_BCs_=true;

    //CYCLIC_BCs_=false;

    this->set_sysLength( (fullLe-1) * ( 1.0 + sin( M_PI / 6 ) ) ); 
    this->set_sysSubLength( (Le-1) * ( 1.0 + sin( M_PI / 6 ) ) );

    //Bz here will be trated as the ratio between phi/phi_0;
    int bc_phase = CYCLIC_BCs_?0:-1;
    //    return  std::polar(1.0,  ( i1 % 2 == 0 ? -1 : 1 ) * peierls_d_ * ( 2 * i1 + sign * 1  ) );

    peierls_d_ = 2.0 * M_PI * this->parameters().Bz_ / double(2.0*(W+bc_phase));


    //print_hamiltonian();

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

    dataP<<H_r.real()/2.7;

    std::cout<<(H_r-H_r.adjoint()).norm()<<std::endl;
  dataP.close();

}



void Graphene_supercell::update_cheb ( type vec[], type p_vec[], type pp_vec[]){

  r_type t = 2.0 * t_standard_/this->a(),
    b_a = 2.0 * this->b()/this->a();

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

      /*
      if( i != 0 )
	vec[n] += t * p_vec[n-1] * peierls(i,-1);
      
      if( i != ( W - 1 ) )
	vec[n] += t * p_vec[n+1] * peierls(i,1);
      
      if( j != ( fullLe - 1 ) && !(j == 0 && i == W-1) && i%2 == 0 )
	vec[n] += t * p_vec[ n + W + 1];
      
      if( j != 0 && i%2 != 0 )
	vec[n] += t * p_vec[ n - W - 1];
      */
      
      
      //if( i != 0 )
      int n2 = j * W + ( i - 1 == -1 ? W - 1 : ( i - 1 )  ) ;
      vec[n] += t * p_vec[ n2 ] * peierls(i,-1);
      
      //if( i != ( W - 1 ) )
      n2 = j * W + ( i + 1 ) % W;
      vec[n] += t * p_vec[ n2 ] * peierls(i,1);
      

      if( i%2 == 0 ){
        n2 = ( ( j + 1 )  % fullLe ) * W + i + 1;
        vec[n] += t * p_vec[ n2 ];
      }
      if(  i%2 != 0 ){
        n2 = ( j - 1 == -1 ? fullLe - 1 : ( j - 1 ) ) * W + i - 1 ;
        vec[n] += t * p_vec[ n2 ];
      }
      

      /*   
      if(!CYCLIC_BCs_ && C==0){      
        if(( i == (W-1) && j==0 ) || (i == 1 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n-1] * peierls(i,-1);

        if( ( i == (W-2) && j==0 ) || (i == 0 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n+1] * peierls(i,1);
      }*/
      
      
      vec[n] *= damp_op[n];
      
      pp_vec[n] = p_vec[n];
      
    }
 } 

 
#pragma omp parallel for 
 for(int n=0; n<LE*W; n++)
   vec[C*W+n] += dis_vec[n] * p_vec[C*W+n] / this->a();

 /* 
 if(CYCLIC_BCs_){
   vertical_BC(2.0, vec,p_vec,damp_op);
   horizontal_BC(2.0, vec,p_vec,damp_op);  
   }
 */
#pragma omp parallel for 
 for(int n=0;n<DIM;n++)
   p_vec[n]  = vec[n];
 
}




void Graphene_supercell::H_ket ( type vec[], type p_vec[]){

  r_type t = t_standard_/this->a(),
    b_a = this->b()/this->a();

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

           
      //if( i != 0 )
      int n2 = j * W + ( i - 1 == -1 ? W - 1 : ( i - 1 )  ) ;
      vec[n] += t * p_vec[ n2 ] * peierls(i,-1);
      
      //if( i != ( W - 1 ) )
      n2 = j * W + ( i + 1 ) % W;
      vec[n] += t * p_vec[ n2 ] * peierls(i,1);
      

      if( i%2 == 0 ){
        n2 = ( ( j + 1 )  % fullLe ) * W + i + 1;
        vec[n] += t * p_vec[ n2 ];
      }
      
      if( i%2 != 0 ){
        n2 = ( j - 1 == -1 ? fullLe - 1 : ( j - 1 ) ) * W + i - 1;
        vec[n] += t * p_vec[ n2 ];
      }
      
      /*
      if( i != 0 )
	vec[n] += t * p_vec[n-1] * peierls(i,-1);
      
      if( i != ( W - 1 ) )
	vec[n] += t * p_vec[n+1] * peierls(i,1);
      
      if( j != ( fullLe - 1 ) && !(j == 0 && i == W-1) && i%2 == 0 )
	vec[n] += t * p_vec[ n + W + 1];
      
      if( j != 0 && i%2 != 0 )
	vec[n] += t * p_vec[ n - W - 1];
      */


      
      /*  
      if(!CYCLIC_BCs_ && C==0){            
        if( (i == (W-1) && j==0) || (i == 1 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n-1] * peierls(i,-1);

        if( (i == (W-2) && j==0) || (i == 0 && j==fullLe-1) )
	  vec[n] -= t * p_vec[n+1] * peierls(i,1);
	  }*/
      
      
      vec[n] *= damp_op[n];      
    }
 } 

#pragma omp parallel for 
 for(int n=0; n<LE*W; n++)
   vec[C*W+n] += dis[n] * p_vec[C*W+n] / this->a();


 /* 
 if(CYCLIC_BCs_){
   vertical_BC(1.0, vec,p_vec,damp_op);
   horizontal_BC(1.0, vec,p_vec,damp_op);  
   }*/
}






void Graphene_supercell::vertical_BC(r_type a2, type vec[], type p_vec[], r_type damp_op[]){
  r_type t   = a2 * t_standard_/this->a();

  int W = this->parameters().W_,
      LE = this->parameters().LE_,
      C = this->parameters().C_;


  const int fullLe = 2*C+LE;

  

  
#pragma omp parallel for 
  for(int j=0; j<fullLe; j++){
    int n_up = j * W +W-1;
    int n_down = j * W;


    vec[n_up]     += damp_op[n_up] * t * p_vec[n_down] * conj(peierls(W,-1));
    vec[n_down]   += damp_op[n_down] * t * p_vec[n_up] * peierls(W,-1);
    
  } 
}



void Graphene_supercell::horizontal_BC(r_type a2, type vec[], type p_vec[], r_type damp_op[]){
  r_type t   = a2 *  t_standard_/this->a();

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
      vec[n_front]   +=  damp_op[n_front] * ( ( n_front )%2==0 ) * t * p_vec[n_back+1];

      if(i>0)
        vec[n_back]    +=  damp_op[n_back]  * ( ( n_back )%2!=0 ) * t * p_vec[n_front-1];
      
    }
}





void Graphene_supercell::vel_op_x (type vec[], type p_vec[] ){
  
  int W   = this->parameters().W_,
      LE  = this->parameters().LE_,
      C  = this->parameters().C_,
      fullLe = ( LE + 2 * C );    
  
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

        int n2 = j * W + ( i - 1 == -1 ? W - 1 : ( i - 1 )  ) ;
	vec[n] += tx2 * ((i%2)==0? -1:1) * p_vec[n2] * peierls(i,-1);
      
        n2 = j * W + ( i + 1 ) % W;  
        vec[n] += tx2 * ((i%2)==0? -1:1) * p_vec[n2] * peierls(i,1);

	
        if(i%2==0 ){
          n2 = ( ( j + 1 )  % fullLe ) * W + i + 1;
	  vec[n] +=  tx1 * p_vec[n2];
	}
      
        if( i%2!=0 ){
          n2 = ( j - 1 == -1 ? fullLe - 1 : ( j - 1 ) ) * W + i - 1;
	  vec[n] += - tx1 * p_vec[n2];
	}

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

  

  /*    
  if(CYCLIC_BCs_){
#pragma omp parallel for 
    for(int j=0; j< ( LE + 2 * C ); j++){
      int n_up = j * W + W-1;
      int n_down = j * W;

      vec[n_up]     +=  tx2 * (((W-1)%2)==0? -1:1) * p_vec[n_down] * conj(peierls(W,-1));
      vec[n_down]   +=  tx2 * ((0%2)==0? -1:1) * p_vec[n_up]   * peierls(W,-1);
    }
 
#pragma omp parallel for 
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (LE + 2 * C - 1) * W + i;
      vec[n_front]   +=  - ( (n_front)%2==0 ) * tx1 * p_vec[n_back+1];
      vec[n_back]    +=    ( (n_back)%2!=0 ) * tx1 * p_vec[n_front-1];
    }
  }
  */

  
};



void Graphene_supercell::vel_op_y (type vec[], type p_vec[] ){
  
  int W   = this->parameters().W_,
      LE  = this->parameters().LE_,
    C  = this->parameters().C_,
    fullLe= LE+2*C;    
  
  r_type dy2 = cos(M_PI/6.0),
         ty2 = dy2 * t_standard_;
  
  std::complex<r_type> Im(0,1.0);


#pragma omp parallel for 
  for(int j=0; j<(LE+2*C); j++){
    for(int i=0; i<W; i++){
      int n =  j * W + i;
      
      vec[n] = 0;

      
      if( n >= C * W ){

        int n2 = j * W + ( i - 1 == -1 ? W - 1 : ( i - 1 )  ) ;
	vec[n] +=  - ty2 * p_vec[n2] * peierls(i,-1);
      
        n2 = j * W + ( i + 1 ) % W;  
	vec[n] += ty2 * p_vec[n2] * peierls(i,1);

      }
    }
  } 

  /*
  if(CYCLIC_BCs_){
#pragma omp parallel for 
    for(int j=0; j< ( LE + 2 * C ); j++){
      int n_up = j * W + W-1;
      int n_down = j * W;

      vec[n_up]     +=  - ty2 * p_vec[n_down]  * conj(peierls(W,-1));
      vec[n_down]   +=  + ty2 * p_vec[n_up]    * peierls(W,-1);
    }

  }
  */
};
