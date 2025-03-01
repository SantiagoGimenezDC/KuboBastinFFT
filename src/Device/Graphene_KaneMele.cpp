#include "Graphene_KaneMele.hpp"
#include<fstream>



Graphene_KaneMele::Graphene_KaneMele(r_type stgr_str, r_type m_str, r_type rashba_str, r_type KM_str, r_type HLD_str, device_vars& parameters): Graphene(parameters), stgr_str_(stgr_str), m_str_(m_str), rashba_str_(rashba_str), KM_str_(KM_str){
    
  this->parameters().DIM_*=4;
  this->parameters().SUBDIM_*=4;


      
  if(this->parameters().C_==0)
    CYCLIC_BCs_=true;


  
  Eigen::Vector3d v1{ this->parameters().W_* 0.5 * sqrt(3.0),   this->parameters().W_* 3.0/2,  0 },
	v2{ -this->parameters().LE_* 0.5 * sqrt(3.0),    this->parameters().LE_ * 3.0 /2,  0 },
	cross_p = v1.cross(v2);
      
  r_type Length = sqrt( abs(cross_p(2)) );


  
  this->set_sysSubLength(Length);
  this->set_sysLength(Length);
      
  //values in eVs;
  r_type t            = t_standard_,
         stgr_str_2   = stgr_str_ ,
         m_str_2      = - m_str_ /4 ,
         rashba_str_2 = - rashba_str_ ,  //Unexplained minus sign here
         KM_str_2     = KM_str_ ,
         HLD_str_2    = HLD_str ;
  

  std::complex<r_type> R_PF    = type(0,1.0) * rashba_str_2 / 3.0 , 
                       KM_PF   = type(0,1.0) * KM_str_2 / (6.0 * sqrt(3.0) ) ,
                       HLD_PF  = type(0,1.0) * HLD_str_2 / (6.0 * sqrt(3.0) ) ;

  Eigen::Vector3d m{0.0,0.0,m_str_2};
  Eigen::Matrix4cd H_bare = Eigen::Matrix4cd::Zero(),
    H_stgr = Eigen::Matrix4d::Zero(),
    H_ex = Eigen::Matrix4d::Zero(),
    H_R_diag = Eigen::Matrix4d::Zero(),
    H_R_off_1 = Eigen::Matrix4d::Zero(),
    H_R_off_2 = Eigen::Matrix4d::Zero();

  H_bare(0,2) = t;
  H_bare(1,3) = t;


  H_stgr.block(0,0,2,2) = stgr_str_2 * Eigen::Matrix2cd::Identity();
  H_stgr.block(2,2,2,2) = - stgr_str_2 * Eigen::Matrix2cd::Identity();


  
  H_ex.block(0,0,2,2) = m[0] * sx + m[1] * sy + m[2] * sz;
  H_ex.block(2,2,2,2) = H_ex.block(0,0,2,2);



  
  H_R_diag.block(0,2,2,2)  = -R_PF * sx;


  H_R_off_1.block(0,2,2,2) = R_PF * ( sx - sqrt(3.0) * sy ) / 2.0;
  H_R_off_2.block(0,2,2,2) = R_PF * ( sx + sqrt(3.0) * sy ) / 2.0;

  /*
  std::cout<<sx<<std::endl;
  std::cout<<sy<<std::endl;
  std::cout<< H_R_diag + H_R_diag.adjoint() <<std::endl;  
  std::cout<< H_R_off_1 <<std::endl;  
  std::cout<< H_R_off_2 <<std::endl;  
  */

  Eigen::Matrix2cd id{{1,0}, {0, 1}};

  
  H_KM_.block(0,0,2,2) = KM_PF * sz ;
  H_KM_.block(2,2,2,2) = -KM_PF * sz ;


  H_HLD_.block(0,0,2,2) = HLD_PF * id ;
  H_HLD_.block(2,2,2,2) = -HLD_PF * id ;


  H_1_ = H_bare + H_stgr + H_ex + H_R_diag ;
  H_2_ = H_bare + H_R_off_1 + H_KM_ + H_HLD_;
  H_3_ = H_bare + H_R_off_2 + H_KM_.adjoint() + H_HLD_.adjoint();
  H_4_ = H_KM_ + H_HLD_;


};


void Graphene_KaneMele::Hk(Eigen::Vector3d k){



};

void Graphene_KaneMele::J (type* ket, type* p_ket, int dir){

  int SUBDIM = this->parameters().SUBDIM_;
  
  int C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;

  
  Eigen::Matrix2cd sx{{0,1},{1,0}}, sy{{0,-type(0,1)}, {type(0,1), 0}}, sz{{1,0}, {0, -1}}, chosen_dir;

  if(dir==0)
    chosen_dir = sx;

  if(dir==1)
    chosen_dir = sy;
  
  if(dir==2)
    chosen_dir = sz;

  

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,SUBDIM),
    eig_p_ket(p_ket, SUBDIM);

    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( ( j + C ) * W + i ) * 4;

      eig_ket.segment(n1,2)   =  chosen_dir * eig_p_ket.segment(n1,2);
      eig_ket.segment(n1+2,2) =  chosen_dir * eig_p_ket.segment(n1+2,2);
 
    }
 }
 
};


void Graphene_KaneMele::print_hamiltonian(){
  int dim = this->parameters().DIM_;
  int DIM = this->parameters().DIM_;

  Eigen::MatrixXcd H_r(dim,dim), S(dim,dim);

  std::ofstream dataP;
  dataP.open("Ham_imag.txt");
  
    for(int j=0;j<dim;j++){
      for(int i=0;i<dim;i++){

	Eigen::Matrix<std::complex<double>, -1, 1>  term_i=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM),
	  tmp=term_i, term_j=term_i, null=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM);

	Eigen::Matrix<double, -1, 1>  null2=Eigen::Matrix<double, -1, 1>::Zero(DIM);
	term_i(i)=1;
	term_j(j)=1;


        //this->update_cheb(tmp.data(),term_j.data(),null.data());
	this->H_ket(tmp.data(),term_j.data());
        //vel_op_x(tmp.data(),term_j.data());
        //vel_op_y(tmp.data(),term_j.data());
	std::complex<double> termy = term_i.dot(tmp);

	 H_r(i,j)=termy;
      }
    }

    dataP<<H_r.imag();

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


void Graphene_KaneMele::projector(type* ket ){

  int DIM = parameters().DIM_;
  Eigen::VectorXcd diag(DIM);

  int option = parameters().projector_option_;
  
  diag.setOnes();
  //option 0 is no projection
  
  if( option == 1 )//spin up
    for( int i = 0; i < DIM; i++ )
      if( i % 2 == 0 )
	diag(i)=0;
  

  if( option == 2 )//spin down
    for( int i = 0; i < DIM; i++ )
      if( (i+1) % 2 == 0 )
	diag(i)=0;


  if( option == 3 )//A sublattice
    for( int i = 0; i < DIM/2; i++ )
      if( i % 2 == 0 ){
	diag(i) = 0;
	diag(i+1) = 0;	
      }


  if( option == 4 )//B sublattice
    for( int i = 0; i < DIM/2; i++ )
      if( (i+1) % 2 == 0 ){
	diag(i) = 0;
	diag(i+1) = 0;	
      }


  Eigen::SparseMatrix<type> sparse_diag_matrix(DIM,DIM);

  sparse_diag_matrix.reserve(Eigen::VectorXi::Constant(DIM, 1));

  for(int i = 0; i < diag.size(); ++i) 
    sparse_diag_matrix.insert(i, i) = diag[i];  // Insert element at (i, i)
  

  
  sparse_diag_matrix.makeCompressed();

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM);
      
  eig_ket = sparse_diag_matrix * eig_ket;

  
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


  r_type b_a = b/a;

          
  Eigen::Matrix4cd Id = Eigen::Matrix4d::Identity(),H_1, H_2, H_3, H_4;

  H_1 = H_1_/a + b_a*Id;
  H_2 = H_2_/a;
  H_3 = H_3_/a;
  H_4 = H_4_/a;


  r_type *dmp_op = damp_op();  
  r_type * disorder_potential = dis();

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim);

    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( j * W + i ) * 4;

      
      eig_ket.segment(n1,4) =  H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      int n2 = (  j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) +=  H_2  * eig_p_ket.segment(n2,4);
      
      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3  * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j + 1 ) % Le ) * W    +    ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_4 * eig_p_ket.segment(n2,4);
      

      
      //Adjoints      
      n2 = ( j * W + ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);
      
      n2 = (  ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W +    ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_4.adjoint() * eig_p_ket.segment(n2,4);
      

      eig_ket.segment(n1,4) *= dmp_op[n1/4];
    }
 }


 
 if( disorder_potential != NULL )
   for( int i = 0; i < 2 * W * Le ; i ++ )
     eig_ket.segment(i,2) += dmp_op[i/2] * disorder_potential[ i ] * eig_p_ket.segment(i,2) / a;
   
};



void Graphene_KaneMele::update_cheb ( type ket[], type p_ket[], type pp_ket[]){
  int Le = this->parameters().LE_,
    W  = this->parameters().W_,
    Dim = this->parameters().DIM_;


  r_type a = this->a(),
         b = this->b();

  r_type b_a = b/a;
  
          
  Eigen::Matrix4cd
    Id = Eigen::Matrix4d::Identity(), H_1, H_2, H_3, H_4;

  H_1 = H_1_/a + b_a*Id;
  H_2 = H_2_/a;
  H_3 = H_3_/a;
  H_4 = H_4_/a;

  r_type *dmp_op = damp_op();  
  r_type * disorder_potential = dis();


  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim),
    eig_pp_ket(pp_ket, Dim);

    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( j * W + i ) * 4;
      
      eig_ket.segment(n1,4) = b_a  * eig_p_ket.segment(n1,4) - 0.5 * dmp_op[n1/4] * eig_pp_ket.segment(n1,4);

      
      eig_ket.segment(n1,4) += H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      
      int n2 = (  j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) +=  H_2  * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3  * eig_p_ket.segment(n2, 4);

      n2 = ( ( ( j + 1 ) % Le ) * W    +    ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_4 * eig_p_ket.segment(n2,4);
      



      //Adjoints
      n2 = ( j * W + ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);

      n2 = (  ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W  +    ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_4.adjoint() * eig_p_ket.segment(n2,4);


      
      eig_ket.segment(n1,4) *= 2.0 * dmp_op[n1/4];
      
      eig_pp_ket.segment(n1,4) = eig_p_ket.segment(n1,4); 
    }
 }


 
 if( disorder_potential != NULL )
   for( int i = 0; i < 2 * W * Le ; i ++ )
     eig_ket.segment(i,2) += dmp_op[i] * disorder_potential[ i ] * eig_p_ket.segment(i,2) / a;

 
 eig_p_ket = eig_ket;
 
};




void Graphene_KaneMele::update_cheb_filtered ( type ket[], type p_ket[], type pp_ket[], type disp_factor){
  int Le = this->parameters().LE_,
    W  = this->parameters().W_,
    Dim = this->parameters().DIM_;


  r_type a = this->a(),
         b = this->b();

  r_type b_a = b/a;
  
          
  Eigen::Matrix4cd
    Id = Eigen::Matrix4d::Identity(), H_1, H_2, H_3, H_4;

  H_1 = H_1_/a + b_a*Id;
  H_2 = H_2_/a;
  H_3 = H_3_/a;
  H_4 = H_4_/a;

  r_type *dmp_op = damp_op();  
  r_type * disorder_potential = dis();


  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim),
    eig_pp_ket(pp_ket, Dim);


#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( j * W + i ) * 4;
      
      eig_ket.segment(n1,4) = b_a  * eig_p_ket.segment(n1,4) - 0.5 * disp_factor * dmp_op[n1/4] * eig_pp_ket.segment(n1,4);

      
      eig_ket.segment(n1,4) += H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      
      int n2 = (  j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) +=  H_2  * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3  * eig_p_ket.segment(n2, 4);

      n2 = ( ( ( j + 1 ) % Le ) * W    +    ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) ) ) * 4;
      eig_ket.segment(n1,4) += H_4 * eig_p_ket.segment(n2,4);
      



      //Adjoints
      n2 = ( j * W + ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);

      n2 = (  ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W +    ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_4.adjoint() * eig_p_ket.segment(n2,4);


      
      eig_ket.segment(n1,4) *= 2.0 * disp_factor * dmp_op[n1/4];
      
      eig_pp_ket.segment(n1,4) = eig_p_ket.segment(n1,4); 
    }
 }


 
 if( disorder_potential != NULL )
   for( int i = 0; i < 2 * W * Le ; i ++ )
     eig_ket.segment(i,2) += dmp_op[i] * disorder_potential[ i ] * eig_p_ket.segment(i,2)/a;

 
 eig_p_ket = eig_ket;
 
};





void Graphene_KaneMele::vel_op_y (type* ket, type* p_ket){

  int Le = this->parameters().LE_,
    W  = this->parameters().W_,
    Dim = this->parameters().DIM_;

  r_type a = this->a(),
    b = this->b(),
    b_a = b/a;
  

  Eigen::Matrix4cd H_KM, H_1, H_2, H_3, H_4;

  H_1 = H_1_;
  H_2 = H_2_;
  H_3 = H_3_;
  H_4 = H_4_;


  //  Eigen::Vector3d v1{ this->parameters().W_* 0.5 * sqrt(3.0),   this->parameters().W_* 3.0/2,  0 },
  //  v2{ -this->parameters().LE_* 0.5 * sqrt(3.0),    this->parameters().LE_ * 3.0 /2,  0 };

  
  std::complex<r_type>
    j1(0, 1.0),
    d_y  = j1, 
    d_y2 = j1 * sin( M_PI / 6.0 ),
    d_y3 = j1 * 3.0 / 2.0;


  
  H_1.block(0,2,2,2) *= -d_y;
  H_1.block(2,0,2,2) *= -d_y;
  H_1.block(0,0,2,2) *= 0.0;
  H_1.block(2,2,2,2) *= 0.0;

  
  
  H_2.block(0,0,2,2) *= d_y3;
  H_2.block(2,2,2,2) *= d_y3;
  
  H_3.block(0,0,2,2) *= d_y3;
  H_3.block(2,2,2,2) *= d_y3;


  
  H_2.block(0,2,2,2) *= d_y2;
  H_2.block(2,0,2,2) *= d_y2;
  
  H_3.block(0,2,2,2) *= d_y2;
  H_3.block(2,0,2,2) *= d_y2;

  
  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim);

    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( (j) * W + i ) * 4;

      
      eig_ket.segment(n1,4) = H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      int n2 = ( ( j ) * W + ( i + 1 ) % W ) * 4;
      //if(i+1<W)
      eig_ket.segment(n1,4) += H_2 * eig_p_ket.segment(n2,4);

      
      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      //if(j+1<Le)
      eig_ket.segment(n1, 4) += H_3 * eig_p_ket.segment(n2, 4);


      
      n2 = ( ( j ) * W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      //if(i-1>=0)
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);

      
      n2 = (  ( ( j - 1 ) == -1 ? (Le - 1) : (j-1) ) * W + i ) * 4;
      //if(j-1>=0)
      eig_ket.segment(n1, 4) +=  H_3.adjoint() * eig_p_ket.segment(n2, 4);
       
    }
 }   


 //eig_ket*=sqrt(3)/3;
 
};



void Graphene_KaneMele::vel_op_x (type* ket, type* p_ket){

  int Le = this->parameters().LE_,
    W  = this->parameters().W_,
    Dim = this->parameters().DIM_;


  r_type a = this->a(),
    b = this->b(),
    b_a = b/a;
  
  Eigen::Vector3d
    v1{ this->parameters().W_* 0.5 * sqrt( 3.0 ),   this->parameters().W_* 3.0 / 2.0,  0 },
    v2{ -this->parameters().LE_* 0.5 * sqrt( 3.0 ),    this->parameters().LE_ * 3.0 / 2.0,  0 };

  
  Eigen::Matrix4cd
    Id = Eigen::Matrix4d::Identity(),
    H_KM,
    H_1,
    H_2,
    H_3,
    H_4;
  
  H_1 = H_1_;
  H_2 = H_2_;
  H_3 = H_3_;
  H_4 = H_4_;
  
  
  std::complex<r_type>
    j1(0, 1.0),
    d_x  = -j1 * cos( M_PI /6.0 ),
    d_x2 = -j1 * 0.5 * sqrt( 3.0 );


    
  H_2.block(0,0,2,2) *= -d_x2;
  H_2.block(2,2,2,2) *= -d_x2;
  
  H_3.block(0,0,2,2) *= d_x2;
  H_3.block(2,2,2,2) *= d_x2;
  
  H_4.block(0,0,2,2) *= 2.0 * d_x2;
  H_4.block(2,2,2,2) *= 2.0 * d_x2;
  

  
  H_2.block(0,2,2,2) *=  -d_x;
  H_2.block(2,0,2,2) *=  -d_x;
  
  H_3.block(0,2,2,2) *=  d_x;
  H_3.block(2,0,2,2) *=  d_x;

  H_4.block(0,2,2,2) *=  d_x;
  H_4.block(2,0,2,2) *=  d_x;
  

  
  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,Dim),
    eig_p_ket(p_ket, Dim);
  
    
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( j * W + i ) * 4;

      

      int n2 = ( j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4)   =  H_2 * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3 * eig_p_ket.segment(n2, 4);

      n2 = ( ( ( j + 1 ) % Le )* W + ( i + 1 ) % W  ) * 4;
      eig_ket.segment(n1,4)  +=  H_4 * eig_p_ket.segment(n2,4);

      

      //Adjoints
      n2 = ( ( j ) * W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      eig_ket.segment( n1, 4 ) +=   H_2.adjoint() * eig_p_ket.segment(n2,4);
      
      n2 = (  ( ( j - 1 ) == -1 ? (Le - 1) : (j-1) ) * W + i ) * 4;
      eig_ket.segment( n1, 4 ) +=  H_3.adjoint() * eig_p_ket.segment(n2, 4);
	
      n2 = ( ( ( j - 1 ) == -1 ? (Le-1) : (j-1) ) * W  + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      eig_ket.segment( n1, 4 ) +=  H_4.adjoint() * eig_p_ket.segment(n2,4);

    }
 }

 //eig_ket *= sqrt(3)/3;
 
};





 




