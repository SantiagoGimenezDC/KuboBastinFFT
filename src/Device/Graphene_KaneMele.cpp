#include "Graphene_KaneMele.hpp"
#include<fstream>

void Graphene_KaneMele::print_hamiltonian(){
  int dim = this->parameters().DIM_;
  int DIM = this->parameters().DIM_;

  Eigen::MatrixXcd H_r(dim,dim), S(dim,dim);

  std::ofstream dataP;
  dataP.open("H_KM.txt");
        
    for(int j=0;j<dim;j++){
      for(int i=0;i<dim;i++){

	Eigen::Matrix<std::complex<double>, -1, 1>  term_i=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM),
	  tmp=term_i, term_j=term_i, null=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM);

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

    dataP<<H_r.real();//.real();

    std::cout<<(H_r-H_r.adjoint()).norm()<<std::endl;
  dataP.close();

  }


void Graphene_KaneMele::traceover(type* traced, type* full_vec, int s, int num_reps){
  int subDim = this->parameters().SUBDIM_,
      C   = this->parameters().C_,
      W   = this->parameters().W_,
      sec_size = subDim/num_reps,
      buffer_length = sec_size;
	
  if( s == num_reps-1 )
      buffer_length += subDim % num_reps;


#pragma omp parallel for 
      for(int i=0;i<buffer_length;i++)
        traced[i] = full_vec[s*sec_size + i+2*C*W];

  };


/*
void Graphene_KaneMele::rearrange_initial_vec(type r_vec[]){ //supe duper hacky
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
    for(int n=0;n<2*Le*W;n++)
      r_vec[2*C*W + n ]=tmp[ n];

}
*/


void Graphene_KaneMele::H_ket (r_type a, r_type b, type* ket, type* p_ket){

  int Le = this->parameters().LE_,
    W  = this->parameters().W_,
    Dim = this->parameters().DIM_;


  r_type t = t_standard_/a,
         b_a = b/a,
         m_str = -m_str_ * t /2,
         rashba_str = rashba_str_ * t,
         KM_str = KM_str_ * t;
  

  std::complex<r_type> R_PF  = -type(0,1.0) * rashba_str / 3.0,    
    KM_PF  = -type(0,1.0) * KM_str / (6.0 * sqrt(3.0) );


  Eigen::Vector3d m{0.0,0.0,m_str};
  Eigen::Matrix4cd H_bare = Eigen::Matrix4cd::Zero(),
    H_ex = Eigen::Matrix4d::Zero(),
    H_R_diag = Eigen::Matrix4d::Zero(),
    H_R_off_1 = Eigen::Matrix4d::Zero(),
    H_R_off_2 = Eigen::Matrix4d::Zero(),
    H_KM = Eigen::Matrix4d::Zero(),
    H_1 = Eigen::Matrix4d::Zero(),
    H_2 = Eigen::Matrix4d::Zero(),
    H_3 = Eigen::Matrix4d::Zero();

  H_bare(0,2) = t;
  H_bare(1,3) = t;

  
  H_bare(0,0) = b_a;
  H_bare(1,1) = b_a;
  H_bare(2,2) = b_a;
  H_bare(3,3) = b_a;


  H_ex.block(0,0,2,2) = m[0] * sx + m[1] * sy + m[2] * sz;
  H_ex.block(2,2,2,2) = H_ex.block(0,0,2,2);

  
  H_R_diag.block(0,2,2,2)  = R_PF * sx; 
  H_R_off_1.block(0,2,2,2) = R_PF * ( sx - sqrt(3.0) * sy ) /2.0;
  H_R_off_2.block(0,2,2,2) = R_PF * ( sx + sqrt(3.0) * sy ) /2.0;

  
  H_KM.block(0,0,2,2) = KM_PF * sz ;
  H_KM.block(2,2,2,2) = -KM_PF * sz ;
  

  H_1 = H_bare + H_ex + H_R_diag + H_KM ;
  H_2 = H_bare + H_R_off_1 + H_KM ;
  H_3 = H_bare + H_R_off_2 + H_KM ;


  

  
  r_type * disorder_potential = dis();

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim);

    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( (j) * W + i ) * 4;

      
      eig_ket.segment(n1,4) = H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      

      int n2 = ( ( j ) * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_2 * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3 * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( j ) * W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);

      n2 = (  ( ( j - 1 ) == -1 ? (Le - 1) : (j-1) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
	




      n2 = ( ( ( j + 1 ) % Le )* W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) ) ) * 4;
      eig_ket.segment(n1,4) += H_KM * eig_p_ket.segment(n2,4);


      n2 = ( ( ( j - 1 ) == -1 ? (W-1) : (j-1) ) * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_KM.adjoint() * eig_p_ket.segment(n2,4);

    }
 }


 
 if( disorder_potential != NULL )
   for( int i = 0; i < W * Le ; i ++ )
     eig_ket.segment(i,4) += disorder_potential[ i ] * eig_p_ket.segment(i,4);
   
};



void Graphene_KaneMele::update_cheb ( type ket[], type p_ket[], type pp_ket[]){
  int Le = this->parameters().LE_,
    W  = this->parameters().W_,
    Dim = this->parameters().DIM_;


  r_type a = this->a(),
         b = this->b();

  
  r_type t = t_standard_/a,
         b_a = b/a,
         m_str = -m_str_ * t/2,
         rashba_str = rashba_str_ * t,
         KM_str = KM_str_ * t;
  

  std::complex<r_type> R_PF  = -type(0,1.0) * rashba_str / 3.0,    
    KM_PF  = -type(0,1.0) * KM_str / (6.0 * sqrt(3.0) );


  Eigen::Vector3d m{0.0,0.0,m_str};
  Eigen::Matrix4cd H_bare = Eigen::Matrix4cd::Zero(),
    H_ex = Eigen::Matrix4d::Zero(),
    H_R_diag = Eigen::Matrix4d::Zero(),
    H_R_off_1 = Eigen::Matrix4d::Zero(),
    H_R_off_2 = Eigen::Matrix4d::Zero(),
    H_KM = Eigen::Matrix4d::Zero(),
    H_1 = Eigen::Matrix4d::Zero(),
    H_2 = Eigen::Matrix4d::Zero(),
    H_3 = Eigen::Matrix4d::Zero();

  H_bare(0,2) = t;
  H_bare(1,3) = t;

  
  H_bare(0,0) = b_a;
  H_bare(1,1) = b_a;
  H_bare(2,2) = b_a;
  H_bare(3,3) = b_a;


  H_ex.block(0,0,2,2) = m[0] * sx + m[1] * sy + m[2] * sz;
  H_ex.block(2,2,2,2) = H_ex.block(0,0,2,2);

  
  H_R_diag.block(0,2,2,2)  = R_PF * sx; 
  H_R_off_1.block(0,2,2,2) = R_PF * ( sx - sqrt(3.0) * sy ) /2.0;
  H_R_off_2.block(0,2,2,2) = R_PF * ( sx + sqrt(3.0) * sy ) /2.0;

  
  H_KM.block(0,0,2,2) = KM_PF * sz ;
  H_KM.block(2,2,2,2) = -KM_PF * sz ;
  

  H_1 = H_bare + H_ex + H_R_diag + H_KM ;
  H_2 = H_bare + H_R_off_1 + H_KM ;
  H_3 = H_bare + H_R_off_2 + H_KM ;
	    
  r_type * disorder_potential = dis();


  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim),
    eig_pp_ket(pp_ket, Dim);

    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( j * W + i ) * 4;
      
      eig_ket.segment(n1,4) = b_a  * eig_p_ket.segment(n1,4) - 0.5 * eig_pp_ket.segment(n1,4);

      
      eig_ket.segment(n1,4) += H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      

      int n2 = (  j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_2 * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3 * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( j ) * W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);

      n2 = (  ( ( j - 1 ) == -1 ? (Le - 1) : (j-1) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
	

      
      n2 = ( ( ( j + 1 ) % Le ) * W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) ) ) * 4;
      eig_ket.segment(n1,4) += H_KM * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j - 1 ) == -1 ? (Le - 1) : (j-1) ) * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_KM.adjoint() * eig_p_ket.segment(n2,4);


      
      eig_ket.segment(n1,4) *= 2.0;
      eig_pp_ket.segment(n1,4) = eig_p_ket.segment(n1,4); 
    }
 }


 
 if( disorder_potential != NULL )
   for( int i = 0; i < W * Le ; i ++ )
     eig_ket.segment(i,4) += disorder_potential[ i ] * eig_p_ket.segment(i,4);

 
 eig_p_ket = eig_ket;
 
};






/*
void Graphene_KaneMele::update_cheb ( type ket[], type p_ket[], type pp_ket[]){

  int fullLe = ( this->parameters().LE_+2*this->parameters().C_ ),
      W      = this->parameters().W_;

  int C = parameters().C_,
      LE = parameters().LE_;

  r_type a = this->a(),
         b = this->b();

  r_type t = t_standard_/a,
         b_a = b/a,
         m_str = m_str_ * t,
         rashba_str = rashba_str_ * t;
  
  std::complex<r_type> f_y  = 2.0 * rashba_str * cos(M_PI/6.0)/3.0,
                       f_x  = 2.0 * rashba_str / 3.0,
                       f_x2 = 2.0 * rashba_str * sin(M_PI/6.0)/3.0;


  r_type * sFilter = damp_op();
  r_type * disorder_potential = dis();
  


  std::complex<r_type> ImUnit(0,1);
  
  f_y *= ImUnit;

  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      ket[ 2 * n ]     =  ( ( b_a - m_str ) * p_ket[ 2 * n ] )    - 0.5 * sFilter[ n ] * pp_ket[ 2 * n ];
      ket[ 2 * n + 1 ] =  ( ( b_a + m_str ) * p_ket[ 2 * n + 1] ) - 0.5 * sFilter[ n ] * pp_ket[ 2 * n + 1 ];

      
      if( i!=0 ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2 * n - 2 ] + (  f_x2 * (((i)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 1 ] ) * peierls(i,-1);
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n - 1 ] + ( -f_x2 * (((i)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 2 ] ) * peierls(i,-1);
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2 * n + 2 ] + (  f_x2 * (((i)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 3 ] ) * peierls(i,1);
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * (((i)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 2 ] ) * peierls(i,1);
      }
      if(j != (fullLe-1) && !(j == 0 && i == W-1) && i%2 == 0 ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2*n+2*(W+1) ]           +  f_x * p_ket[ 2 * n + 2 * (W+1) + 1 ] ) ;
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n + 2 * (W+1) + 1]  -  f_x * p_ket[ 2 * n + 2 * (W+1) ]);
      }
      if(j != 0  && i%2 != 0 ){
	ket[ 2 * n ]     +=  ( - t * p_ket[ 2 * n - 2 * ( W + 1 ) ]     -  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ] );
	ket[ 2 * n + 1 ] +=  ( - t * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ] +  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) ]   );
      }
      
      ket[ 2 * n ]     *= 2.0 * sFilter[ n ];
      ket[ 2 * n + 1 ] *= 2.0 * sFilter[ n ];
      
    }
 }
 
   if( disorder_potential != NULL ){
     for( int i = W * C; i < W * C + W * LE ; i ++ ){
       ket[ 2 * i ]      +=  2 * sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i ] / a;
       ket[ 2 * i + 1 ]  +=  2 * sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i +1 ] / a;
     }    
   }
 


 
 if(CYCLIC_BCs_){
   //if(W%2!=0)
   // std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;


     for(int j=0; j<fullLe; j++){

       int n_up = j * W + W - 1;
       int n_down = j * W;


       ket[ 2 * n_up ]    += 2.0 * sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * (((0)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n_down + 1] ) * peierls(W-1,1);
       ket[ 2 * n_up + 1] += 2.0 * sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down + 1] + (  f_x2 * (((0)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n_down ]  ) * peierls(W-1,1);
      
      
       ket[ 2 * n_down]      += 2.0 * sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up ] + ( -f_x2 * ( ((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up + 1 ]   ) * peierls(0,-1);
       ket[ 2 * n_down + 1 ] += 2.0 * sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up + 1 ] + ( f_x2 * (((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up ]   ) * peierls(0,-1);
      
     } 


     //if(fullLe%2!=0)
     //std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 
   for(int i=0; i<W; i++){
     int n_front = i;
     int n_back = (fullLe-1) * W + i;

      
     ket[ 2 * n_front ]     += 2.0 * sFilter[ n_front ] * ((n_front)%2 == 0) * ( -t * p_ket[ 2 * (n_back+1) ]    + ( -f_x * (((n_front)%2)==0? -1.0:1.0) ) * p_ket[ 2 * (n_back+1) + 1 ] );
     ket[ 2 * n_front + 1 ] += 2.0 * sFilter[ n_front ] * ((n_front)%2 == 0) *( -t * p_ket[ 2 * (n_back+1) + 1 ] + ( f_x * (((n_front)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_back+1) ]  );


     ket[ 2 * n_back ]     += 2.0 * sFilter[ n_back ] * ((n_back)%2 != 0) * ( -t * p_ket[ 2 * (n_front-1) ]     + ( f_x * (((n_back)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_front-1) + 1 ]   );
     ket[ 2 * n_back + 1 ] += 2.0 * sFilter[ n_back ] * ((n_back)%2 != 0) * ( -t * p_ket[ 2 * (n_front-1) + 1 ] + ( -f_x * (((n_back)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_front-1) ]   );
    }

 }

#pragma omp parallel for 
 for(int n=0;n<2*fullLe*W;n++){
   pp_ket[n] = p_ket[n];
   p_ket[n]  = ket[n];
 }
 
}; 
*/



void Graphene_KaneMele::vel_op_x (type* ket, type* p_ket){

  int Le = this->parameters().LE_,
    W  = this->parameters().W_,
    Dim = this->parameters().DIM_;

  r_type a = this->a(),
         b = this->b();

  
  r_type t = t_standard_/a,
         b_a = b/a,
         m_str = -m_str_ * t /2,
         rashba_str = rashba_str_ * t,
         KM_str = KM_str_ * t;
  

  std::complex<r_type> R_PF  = -type(0,1.0) * rashba_str / 3.0,    
    KM_PF  = -type(0,1.0) * KM_str / (6.0 * sqrt(3.0) );

  std::complex<r_type>
    d_x  = -a0_ * sin( M_PI /6.0 ),
    d_x2 = a0_ * sin( M_PI /6.0 ),
    d_x3 = 2.0 * a0_ * sin( M_PI /6.0 );


  Eigen::Vector3d m{0.0,0.0,m_str};
  Eigen::Matrix4cd H_bare = Eigen::Matrix4cd::Zero(),
    H_ex = Eigen::Matrix4d::Zero(),
    H_R_diag = Eigen::Matrix4d::Zero(),
    H_R_off_1 = Eigen::Matrix4d::Zero(),
    H_R_off_2 = Eigen::Matrix4d::Zero(),
    H_KM = Eigen::Matrix4d::Zero(),
    H_1 = Eigen::Matrix4d::Zero(),
    H_2 = Eigen::Matrix4d::Zero(),
    H_3 = Eigen::Matrix4d::Zero();

  H_bare(0,2) = t;
  H_bare(1,3) = t;

  
  H_bare(0,0) = b_a;
  H_bare(1,1) = b_a;
  H_bare(2,2) = b_a;
  H_bare(3,3) = b_a;


  H_ex.block(0,0,2,2) = m[0] * sx + m[1] * sy + m[2] * sz;
  H_ex.block(2,2,2,2) = H_ex.block(0,0,2,2);

  
  H_R_diag.block(0,2,2,2)  = R_PF * sx; 
  H_R_off_1.block(0,2,2,2) = R_PF * ( sx - sqrt(3.0) * sy ) /2.0;
  H_R_off_2.block(0,2,2,2) = R_PF * ( sx + sqrt(3.0) * sy ) /2.0;

  
  H_KM.block(0,0,2,2) = KM_PF * sz ;
  H_KM.block(2,2,2,2) = -KM_PF * sz ;
  

  H_1 = H_bare + H_ex + H_R_diag + H_KM ;
  H_2 = H_bare + H_R_off_1 + H_KM ;
  H_3 = H_bare + H_R_off_2 + H_KM ;


  

  
  r_type * disorder_potential = dis();

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim);

    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( j * W + i ) * 4;

      

      int n2 = ( j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) = d_x2 * H_2 * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += d_x * H_3 * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( j ) * W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      eig_ket.segment(n1,4) += - d_x2 * H_2.adjoint() * eig_p_ket.segment(n2,4);

      n2 = (  ( ( j - 1 ) == -1 ? (Le - 1) : (j-1) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += -d_x * H_3.adjoint() * eig_p_ket.segment(n2, 4);
	




      n2 = ( ( ( j + 1 ) % Le )* W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) ) ) * 4;
      eig_ket.segment(n1,4) += H_KM * eig_p_ket.segment(n2,4);


      n2 = ( ( ( j - 1 ) == -1 ? (W-1) : (j-1) ) * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_KM.adjoint() * eig_p_ket.segment(n2,4);

    }
 }


 
 if( disorder_potential != NULL )
   for( int i = 0; i < W * Le ; i ++ )
     eig_ket.segment(i,4) += disorder_potential[ i ] * eig_p_ket.segment(i,4);
   
};





 




