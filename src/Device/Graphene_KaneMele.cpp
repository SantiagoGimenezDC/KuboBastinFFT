#include "Graphene_KaneMele.hpp"
#include<fstream>
#include <eigen3/Eigen/Eigenvalues>
#include<fftw3.h>
#include<omp.h>
//#include<fftw_threads.h>



Graphene_KaneMele::Graphene_KaneMele(int is_k_space, r_type range, r_type stgr_str, r_type m_str, r_type rashba_str, r_type KM_str, r_type HLD_str, device_vars& parameters): Graphene(parameters), stgr_str_(stgr_str), m_str_(m_str), rashba_str_(rashba_str), KM_str_(KM_str){
    
  this->parameters().DIM_*=4;
  this->parameters().SUBDIM_*=4;


      
  if(this->parameters().C_==0)
    CYCLIC_BCs_=true;

  

  range_ = range;

  if( is_k_space == 1)
   k_space_ = true;
  else
   k_space_ = false;

  Eigen::Vector3d v1{ double(this->parameters().W_)* 0.5 * sqrt(3.0),   double(this->parameters().W_)* 3.0/2,  0 },
    v2{ -double(this->parameters().LE_)* 0.5 * sqrt(3.0),    double(this->parameters().LE_) * 3.0 /2,  0 },
	cross_p = v1.cross(v2);
      
  r_type Length = a0_ * sqrt( abs(cross_p(2))  );


  
  this->set_sysSubLength(Length);
  this->set_sysLength(Length);
      

  //values in eVs;
  r_type t            = t_standard_,
         stgr_str_2   = stgr_str_ ,
         m_str_2      = m_str_ ,
         rashba_str_2 = rashba_str_ ,
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



  

  std::complex<double> I(0, 1);  // Imaginary unit

  
  H_k0_bare_(0, 2) = t_standard_; //removed the - sign???
  H_k0_bare_(1, 3) = t_standard_;
  
  H_k0_ex_ =  H_ex; // should be a This 4.0 compensates for the /4 in the PF

  H_k0_R_1_.block(0,2,2,2) = -R_PF * sx; //Compensating for the unexplained - sign; same on both next lines.
  H_k0_R_2_.block(0,2,2,2) = R_PF * (sx - sqrt3_ * sy) / 2.0;
  H_k0_R_3_.block(0,2,2,2) = R_PF * (sx + sqrt3_ * sy) / 2.0;
    
  H_k0_KM_.block( 0, 0, 2, 2 ) =  ( KM_PF * sz  );
  H_k0_KM_.block( 2, 2, 2, 2 ) = -( KM_PF * sz  );


  if(k_space_)
    this->build_Hk();


  

  size_t W = this->parameters().W_;
  size_t Le = this->parameters().LE_;
 
  int fftw_threads_init(void);
    
  void fftw_plan_with_nthreads(int nthreads = omp_get_max_threads());

  fft_input_  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * W * Le );
  fft_output_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * W * Le );

  fftw_plan_FORWD_= fftw_plan_dft_2d(W, Le, fft_input_, fft_output_, FFTW_FORWARD, FFTW_ESTIMATE),
  fftw_plan_BACK_ = fftw_plan_dft_2d(W, Le, fft_input_, fft_output_, FFTW_BACKWARD, FFTW_ESTIMATE);// 
  
};




void Graphene_KaneMele::build_Hk(){
  size_t W = parameters().W_;
  size_t Le = parameters().LE_;
  size_t subdim = 0;
    

  std::vector<Eigen::Vector2d> nonZeroList;
  
  Eigen::Vector2d  dirac_1 = 2.0 * M_PI / ( 3.0 * a_ ) * Eigen::Vector2d(  1, 3.0/sqrt(3.0) );
  Eigen::Vector2d  dirac_2 = 2.0 * M_PI / ( 3.0 * a_ ) * Eigen::Vector2d( -1, 3.0/sqrt(3.0) );

  
  //#pragma omp parallel for
  for(size_t j=0; j<Le; j++){
    for(size_t i=0; i<W; i++){

      
      double kx = ( i * b1_(0) + j * b2_(0) ) / W,
             ky = ( i * b1_(1) + j * b2_(1) ) / Le;
      
      Eigen::Vector2d k_vec = Eigen::Vector2d(kx, ky);
    
      double dist_1 = ( k_vec - dirac_1 ).norm(),
	     dist_2 = ( k_vec - dirac_2 ).norm();

      if( dist_1 < range_ * b1_.norm() || dist_2 < range_ * b1_.norm() ){
	subdim++;
	nonZeroList_.push_back(Eigen::Vector2d(i, j));
      }
    }
  }



    size_t DIM = parameters().DIM_;

    H_k_cut_.resize(4, subdim*4);
    v_k_x_cut_.resize(4, subdim*4);
    v_k_y_cut_.resize(4, subdim*4);

    projector_.resize(DIM);
    projector_ = Eigen::VectorXcd::Zero(DIM);

    
    for (size_t i = 0; i < nonZeroList_.size(); i++) {
      const auto& vec = nonZeroList_[i]; 
      size_t i2 = vec(0);
      size_t j2 = vec(1);

      projector_.segment( (i2 + j2 * W ) * 4, 4 ) = Eigen::VectorXcd::Constant( 4, 1.0 );
      H_k_cut_.block(0, i * 4, 4, 4) = this->Hk_single(Eigen::Vector2d(i2,j2));

      Eigen::MatrixXcd vk = vk_single( Eigen::Vector2d(i2,j2) );
      v_k_x_cut_.block(0, i * 4, 4, 4) = vk.block(0,0,4,4);
      v_k_y_cut_.block(0, i * 4, 4, 4) = vk.block(0,4,4,4);
  }



  parameters().SUBDIM_ = 4 * subdim;


  

  Eigen::Vector3d v1{ double(this->parameters().W_* 0.5 * sqrt(3.0) ),   double(this->parameters().W_) * 3.0/2.0,  0 },
    v2{ -double(this->parameters().LE_* 0.5 * sqrt(3.0)),    double(this->parameters().LE_) * 3.0 /2.0,  0 },
	          cross_p = v1.cross(v2);
      
  r_type Length = a0_ * sqrt( abs(cross_p(2)) );


  if(this->parameters().dis_str_ == 0.0){
    parameters().DIM_ = 4 * subdim;

  }
  else{
    DIS_DIM_ = this->parameters().DIM_;
    eig_ket_re_.resize(DIS_DIM_);
    eig_ket_re_sub_.resize(parameters().SUBDIM_);
    //    Length *= sqrt( double( this->parameters().SUBDIM_ ) / double( this->parameters().DIM_ ) );
    parameters().DIM_ = 4 * subdim; //TO USE cut_1, just erase this.
  }

  



  
  this->set_sysSubLength(Length);
  this->set_sysLength(Length);

}



void Graphene_KaneMele::diagonalize_kSpace(){

  int W = parameters().W_;
  int Le = parameters().LE_;
  int subdim = 0;
    
  eigenvalues_k_.resize( W * Le * 4);
  U_k_.resize(4, W*Le*4);
  H_k_.resize(4, W*Le*4);

  v_k_x_.resize(4, W*Le*4);
  v_k_y_.resize(4, W*Le*4);


  std::vector<Eigen::Vector2d> nonZeroList;
  
  Eigen::Vector2d  dirac_1 = 2.0 * M_PI / ( 3.0 * a_ ) * Eigen::Vector2d(  1, 3.0/sqrt(3.0) );
  Eigen::Vector2d  dirac_2 = 2.0 * M_PI / ( 3.0 * a_ ) * Eigen::Vector2d( -1, 3.0/sqrt(3.0) );

  /*
    k_points_2 = np.zeros((grid_size*grid_size,2))
        
    for y in range(grid_size ):
        for x in range(grid_size):
            new_vec = x * b1 / grid_size + y * b2 / grid_size
            dist1 = np.abs(new_vec - dirac_1)
            dist2 = np.abs(new_vec - dirac_2)
  */
		
  //#pragma omp parallel for
  for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){

      
      double kx = ( i * b1_(0) + j * b2_(0) ) / W,
             ky = ( i * b1_(1) + j * b2_(1) ) / Le;
      
      Eigen::Vector2d k_vec = Eigen::Vector2d(kx, ky);
    
      double dist_1 = (k_vec - dirac_1).norm(),
	dist_2 = (k_vec - dirac_2).norm(),
	nullify=0.0;

      




      if( dist_1 < range_ * b1_.norm() || dist_2 < range_* b1_.norm() ){
        nullify = 1.0;
	subdim++;
      }
	
      H_k_.block( 0, ( i + j * W ) * 4, 4, 4 ) = nullify * this->Hk_single(Eigen::Vector2d(i,j));
      
      eigenSol sol = Uk_single(Eigen::Vector2d(i,j));
      eigenvalues_k_.segment((i+j*W)*4, 4) = nullify * sol.eigenvalues_;
      U_k_.block( 0, (i+j*W)*4, 4, 4 ) = nullify * sol.Uk_ ;
      H_k_.block( 0, (i+j*W)*4, 4, 4 ) = nullify * this->Hk_single(Eigen::Vector2d(i,j));
      
      //       std::cout<< sol.Uk_.adjoint()   *  this->Hk_single(Eigen::Vector2d(i,j))   *   sol.Uk_  <<std::endl<<std::endl<<std::endl;
      //       std::cout<< this->Hk_single(Eigen::Vector2d(i,j))  <<std::endl<<std::endl<<std::endl;

      
      
      Eigen::MatrixXcd vk = vk_single( Eigen::Vector2d(i,j) );
      v_k_x_.block( 0, (i+j*W)*4, 4, 4 ) = nullify*vk.block(0,0,4,4);
      v_k_y_.block( 0, (i+j*W)*4, 4, 4 ) = nullify*vk.block(0,4,4,4);

    }
  }

  parameters().SUBDIM_ = 4 * subdim;

    
};







void Graphene_KaneMele::k_vel_op_x (type* ket, type* p_ket){

  size_t W = parameters().W_,
    Le = parameters().LE_,
    SUBDIM = parameters().SUBDIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,SUBDIM),
    eig_p_ket(p_ket, SUBDIM);
  
  #pragma omp parallel for
  for(size_t j=0; j<Le; j++)
    for(size_t i=0; i<W; i++)
      //Eigen::Matrix4cd H_band = eigenvalues_k_.segment( ( i + j * W ) * 4, 4 ).asDiagonal(); //( H_band - b * Id ) / a   *   eig_p_ket.segment( ( i + j * W ) * 4, 4 ); //
      eig_ket.segment( ( i + j * W ) * 4, 4 ) = v_k_x_.block( 0, ( i + j * W ) * 4, 4, 4 )   *   eig_p_ket.segment( ( i + j * W ) * 4, 4 );
      
   
};




void Graphene_KaneMele::k_vel_op_y (type* ket, type* p_ket){

  size_t W = parameters().W_,
    Le = parameters().LE_,
    SUBDIM = parameters().SUBDIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,SUBDIM),
    eig_p_ket(p_ket, SUBDIM);

  
  #pragma omp parallel for
  for(size_t j=0; j<Le; j++)
    for(size_t i=0; i<W; i++)
      //Eigen::Matrix4cd H_band = eigenvalues_k_.segment( ( i + j * W ) * 4, 4 ).asDiagonal(); //( H_band - b * Id ) / a   *   eig_p_ket.segment( ( i + j * W ) * 4, 4 ); //
      eig_ket.segment( ( i + j * W ) * 4, 4 ) = v_k_y_.block( 0, ( i + j * W ) * 4, 4, 4 )   *   eig_p_ket.segment( ( i + j * W ) * 4, 4 );
      
   
};


void Graphene_KaneMele::Hk_ket (r_type a, r_type b, type* ket, type* p_ket){

  size_t W = parameters().W_,
    Le = parameters().LE_,
    DIM = parameters().DIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM);

  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 

  
  #pragma omp parallel for
  for(size_t j=0; j<Le; j++)
    for(size_t i=0; i<W; i++)
      //Eigen::Matrix4cd H_band = eigenvalues_k_.segment( ( i + j * W ) * 4, 4 ).asDiagonal(); //( H_band - b * Id ) / a   *   eig_p_ket.segment( ( i + j * W ) * 4, 4 ); //
      eig_ket.segment( ( i + j * W ) * 4, 4 ) = ( H_k_.block( 0, ( i + j * W ) * 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( ( i + j * W ) * 4, 4 );
      
   
};









void Graphene_KaneMele::Hk_update_cheb ( type ket[], type p_ket[], type pp_ket[]){
  size_t W = parameters().W_,
    Le = parameters().LE_,
    DIM = parameters().DIM_;

  double a = this->a(),
         b = this->b();
    
  
  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM),
    eig_pp_ket(pp_ket, DIM);

  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 


  #pragma omp parallel for
  for(size_t j=0; j<Le; j++)
    for(size_t i=0; i<W; i++){
      //Eigen::Matrix4cd H_band = eigenvalues_k_.segment( ( i + j * W ) * 4, 4 ).asDiagonal(); //
      eig_ket.segment( ( i + j * W ) * 4, 4 ) =  2 * ( H_k_.block( 0, ( i + j * W ) * 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( ( i + j * W ) * 4, 4 )  -   eig_pp_ket.segment( ( i + j * W ) * 4, 4 );
    }
  

  eig_pp_ket = eig_p_ket;
  eig_p_ket = eig_ket;
   
};







void Graphene_KaneMele::Hk_ket_cut (r_type a, r_type b, type* ket, type* p_ket){

  size_t W = parameters().W_,
    DIM = parameters().DIM_;


  
  r_type * disorder_potential = dis();
  //  Eigen::Map<Eigen::VectorXd>dis(disorder_potential,DIM);

  
  Eigen::Map<Eigen::VectorXcd>
    eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM);





  //eig_ket_re = projector_.asDiagonal() * eig_ket_re;
  
    #pragma omp parallel for
    for(size_t i = 0; i < DIM; i++)
      eig_ket_re_[i] = projector_[i] * eig_p_ket[i] ;

  
    to_kSpace(eig_ket_re_.data(), eig_ket_re_.data(), 1);  

    #pragma omp parallel for
    for(size_t i = 0; i < DIM; i++)
      eig_ket_re_[i] = disorder_potential[i] * eig_ket_re_[i] / a;

    to_kSpace(eig_ket_re_.data(), eig_ket_re_.data(), -1);

    #pragma omp parallel for
    for(size_t i = 0; i < DIM; i++)
      eig_ket_re_[i] = projector_[i] * eig_ket_re_[i];
    
    //   eig_ket_re = projector_.asDiagonal() * eig_ket_re;
    
 
  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 

#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++) {
    const auto& vec = nonZeroList_[i]; 
    size_t i2 = vec(0);
    size_t j2 = vec(1);
      eig_ket.segment( ( i2 + j2 * W ) * 4, 4 ) = ( H_k_cut_.block( 0, i* 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( ( i2 + j2 * W ) * 4, 4 );
  }

  eig_ket += eig_ket_re_;
  
};




void Graphene_KaneMele::Hk_ket_cut_2 (r_type a, r_type b, type* ket, type* p_ket){

  size_t SUBDIM = parameters().SUBDIM_;
  r_type * disorder_potential = dis();


  Eigen::Map<Eigen::VectorXcd>
    eig_ket(ket,SUBDIM),
    eig_p_ket(p_ket, SUBDIM);
  
  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 

  eig_ket_re_.setZero();
  to_rSpace_pruned_2(eig_ket_re_.data(), eig_p_ket.data());  

  #pragma omp parallel for
  for(size_t i = 0; i < DIS_DIM_/2; i++){
    eig_ket_re_[2*i] = disorder_potential[i] * eig_ket_re_[2*i] / a;
    eig_ket_re_[2*i+1] = disorder_potential[i] * eig_ket_re_[2*i+1] / a;
  }
  to_kSpace_pruned_2(eig_ket_re_sub_.data(), eig_ket_re_.data());


  
#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++) 
      eig_ket.segment( i * 4, 4 ) = ( H_k_cut_.block( 0, i* 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( i * 4, 4 );

  eig_ket += eig_ket_re_sub_;
  
};




void Graphene_KaneMele::Hk_ket_cut_clean (r_type a, r_type b, type* ket, type* p_ket){

  size_t DIM = parameters().DIM_;


  Eigen::Map<Eigen::VectorXcd>
    eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM);
 
  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 

#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++) 
      eig_ket.segment( i * 4, 4 ) = ( H_k_cut_.block( 0, i* 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( i * 4, 4 );

};





void Graphene_KaneMele::Hk_update_cheb_cut (type* ket, type* p_ket, type* pp_ket){

  size_t W = parameters().W_,
    DIM = parameters().DIM_;

  r_type a = this->a();
  r_type b = this->b(); 


  r_type * disorder_potential = dis();
  Eigen::Map<Eigen::VectorXd>dis(disorder_potential,DIM);


  
  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM),
    eig_pp_ket(pp_ket, DIM);




  //eig_ket_re = projector_.asDiagonal() * eig_ket_re;
  
    #pragma omp parallel for
    for(size_t i = 0; i < DIM; i++)
      eig_ket_re_[i] = projector_[i] * eig_p_ket[i] ;

  
    to_kSpace(eig_ket_re_.data(), eig_ket_re_.data(), 1);  

    #pragma omp parallel for
    for(size_t i = 0; i < DIM; i++)
      eig_ket_re_[i] = disorder_potential[i] * eig_ket_re_[i] / a;
    //eig_ket_re = dis.asDiagonal() * eig_ket_re / a;
 
    to_kSpace(eig_ket_re_.data(), eig_ket_re_.data(), -1);

    #pragma omp parallel for
    for(size_t i = 0; i < DIM; i++)
      eig_ket_re_[i] = projector_[i] * eig_ket_re_[i];
  
    //   eig_ket_re = projector_.asDiagonal() * eig_ket_re;


    /*
  if( this->parameters().dis_str_ != 0.0 ){
    eig_ket_re = projector_.asDiagonal() * eig_ket_re;
  
    to_kSpace(eig_ket_re.data(), eig_ket_re.data(), 1);  
    eig_ket_re = dis.asDiagonal() * eig_ket_re / a;
    to_kSpace(eig_ket_re.data(), eig_ket_re.data(), -1);

    eig_ket_re = projector_.asDiagonal() * eig_ket_re;
    }*/
  
  
  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 

#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++) {
    const auto& vec = nonZeroList_[i]; 
    size_t i2 = vec(0);
    size_t j2 = vec(1); 
    eig_ket.segment( ( i2 + j2 * W ) * 4, 4 ) = 2.0 * ( H_k_cut_.block( 0, i * 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( ( i2 + j2 * W ) * 4, 4 )  -  eig_pp_ket.segment( ( i2 + j2 * W ) * 4, 4 );
  }

  
  eig_ket += 2.0 * eig_ket_re_;

  
  eig_pp_ket = eig_p_ket;
  eig_p_ket = eig_ket;
 
};





void Graphene_KaneMele::Hk_update_cheb_cut_2 (type* ket, type* p_ket, type* pp_ket){

  size_t SUBDIM = parameters().SUBDIM_;
  r_type * disorder_potential = dis();
  r_type a = this->a();
  r_type b = this->b(); 


  Eigen::Map<Eigen::VectorXcd>
    eig_ket(ket,SUBDIM),
    eig_p_ket(p_ket, SUBDIM),
    eig_pp_ket(pp_ket, SUBDIM);
  
  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 

  eig_ket_re_.setZero();
  to_rSpace_pruned_2(eig_ket_re_.data(), eig_p_ket.data());  

  #pragma omp parallel for
  for(size_t i = 0; i < DIS_DIM_/2; i++){
    eig_ket_re_[2*i] = disorder_potential[i] * eig_ket_re_[2*i] / a;
    eig_ket_re_[2*i+1] = disorder_potential[i] * eig_ket_re_[2*i+1] / a;
  }

  to_kSpace_pruned_2(eig_ket_re_sub_.data(), eig_ket_re_.data());




  
#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++) 
      eig_ket.segment( i * 4, 4 ) = 2.0 * ( H_k_cut_.block( 0, i* 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( i * 4, 4 ) - eig_pp_ket.segment( i * 4, 4 ) ;

   
  eig_ket += 2.0 * eig_ket_re_sub_;

  
  eig_pp_ket = eig_p_ket;
  eig_p_ket = eig_ket;
 
};




void Graphene_KaneMele::Hk_update_cheb_cut_clean (type* ket, type* p_ket, type* pp_ket){

  size_t W = parameters().W_,
    DIM = parameters().DIM_;

  r_type a = this->a();
  r_type b = this->b(); 
  
  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM),
    eig_pp_ket(pp_ket, DIM);


  Eigen::Matrix4cd Id = Eigen::Matrix4cd::Zero(); 

#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++)
    eig_ket.segment( i * 4, 4 ) = 2.0 * ( H_k_cut_.block( 0, i * 4, 4, 4 ) - b * Id ) / a   *   eig_p_ket.segment( i * 4, 4 )  -  eig_pp_ket.segment( i * 4, 4 );

  
  eig_pp_ket = eig_p_ket;
  eig_p_ket = eig_ket;
 
};




void Graphene_KaneMele::k_vel_op_x_cut (type* ket, type* p_ket){

  size_t W = parameters().W_,
    DIM = parameters().DIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM);



#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++) {
    const auto& vec = nonZeroList_[i]; 
    size_t i2 = vec(0);
    size_t j2 = vec(1); 
    eig_ket.segment( ( i2 + j2 * W ) * 4, 4 ) = v_k_x_cut_.block( 0, i* 4, 4, 4 )  *   eig_p_ket.segment( ( i2 + j2 * W ) * 4, 4 );
  }
 
};


void Graphene_KaneMele::k_vel_op_x_cut_clean (type* ket, type* p_ket){

  size_t W = parameters().W_,
    DIM = parameters().DIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM);



#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++)
    eig_ket.segment( i * 4, 4 ) = v_k_x_cut_.block( 0, i* 4, 4, 4 )  *   eig_p_ket.segment( i * 4, 4 );
  
 
};



void Graphene_KaneMele::k_vel_op_y_cut (type* ket, type* p_ket){

  size_t W = parameters().W_,
    DIM = parameters().DIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM);

  


#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++) {
    const auto& vec = nonZeroList_[i]; 
    int i2 = vec(0);
    int j2 = vec(1); 
    eig_ket.segment( ( i2 + j2 * W ) * 4, 4 ) = v_k_y_cut_.block( 0, i * 4, 4, 4 )   *   eig_p_ket.segment( ( i2 + j2 * W ) * 4, 4 );
  }
 
};




void Graphene_KaneMele::k_vel_op_y_cut_clean (type* ket, type* p_ket){

  size_t DIM = parameters().DIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,DIM),
    eig_p_ket(p_ket, DIM);


#pragma omp parallel for  
  for (size_t i = 0; i < nonZeroList_.size(); i++)
    eig_ket.segment( i * 4, 4 ) = v_k_y_cut_.block( 0, i * 4, 4, 4 )   *   eig_p_ket.segment( i * 4, 4 );
  
 
};



















void Graphene_KaneMele::Uk_ket ( type* ket, type* p_ket){

  int W = parameters().W_,
    Le = parameters().LE_,
    SUBDIM = parameters().SUBDIM_;

  Eigen::Map<Eigen::VectorXcd> eig_ket(ket,SUBDIM),
    eig_p_ket(p_ket, SUBDIM);


  #pragma omp parallel for
  for(int j=0; j<Le; j++)
    for(int i=0; i<W; i++)
      eig_ket.segment( ( i + j * W ) * 4, 4 ) =  U_k_.block( 0, ( i + j * W ) * 4, 4, 4 )    *   eig_p_ket.segment( ( i + j * W ) * 4, 4 );
    
};


eigenSol Graphene_KaneMele::Uk_single(Eigen::Vector2d k) {

  Eigen::Matrix4cd H_k = this->Hk_single(k), U_k;
  eigenSol solution;


  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> solver(H_k);

  solution.eigenvalues_ = solver.eigenvalues();
  U_k = solver.eigenvectors();

  // for(int i = 0; i<4;i++)
  //  U_k.col(i).normalize();
  
  solution.Uk_ = U_k;

  // std::cout<< U_k.col(0).norm()<<" "<<U_k.col(1).norm()<<"  "<< U_k.col(2).norm() <<std::endl<<std::endl;  
  // std::cout<< solution.Uk_.adjoint()   *  H_k   *   solution.Uk_  <<std::endl<<std::endl<<std::endl;  
  
  return solution;
};



Eigen::Matrix4cd Graphene_KaneMele::Hk_single(Eigen::Vector2d k) {
    
    std::complex<double> I(0, 1);  // Imaginary unit

    size_t W  = parameters().W_;
    size_t Le = parameters().LE_;
    
    // Components of k-vector
    double kx = ( k(0) * b1_(0) + k(1) * b2_(0) ) / W,
           ky = ( k(0) * b1_(1) + k(1) * b2_(1) ) / Le;

    // Nearest-neighbor hopping phase factors
    std::complex<double> gamma1 = exp( I * ( kx * d1_(0) + ky * d1_(1) ) );
    std::complex<double> gamma2 = exp( I * ( kx * d2_(0) + ky * d2_(1) ) );
    std::complex<double> gamma3 = exp( I * ( kx * d3_(0) + ky * d3_(1) ) );

    std::complex<double> gamma4 = exp( I * ( kx * d4_(0) + ky * d4_(1) ) );
    std::complex<double> gamma5 = exp( I * ( kx * d5_(0) + ky * d5_(1) ) );
    std::complex<double> gamma6 = exp( I * ( kx * d6_(0) + ky * d6_(1) ) );

    
    
    // Hamiltonian H_k
    Eigen::Matrix4cd H_k   = Eigen::Matrix4cd::Zero(),
                     H_k_R = Eigen::Matrix4cd::Zero();

    H_k_R  = H_k0_R_1_ * gamma3;
    H_k_R += H_k0_R_2_ * gamma1;
    H_k_R += H_k0_R_3_ * gamma2; 

    H_k  = H_k0_ex_;
    H_k += H_k0_bare_ * ( gamma1 + gamma2 + gamma3 );
    H_k += H_k_R;
    H_k += H_k0_KM_   * ( gamma4 + gamma5 + gamma6 );

    
    
    Eigen::Matrix4cd H_k_adj = H_k.adjoint(); 
    H_k = H_k  + H_k_adj; 


    
    return H_k;  
}




Eigen::MatrixXcd Graphene_KaneMele::vk_single(Eigen::Vector2d k) {
    
    std::complex<double> I(0, 1);  // Imaginary unit

    size_t W  = parameters().W_;
    size_t Le = parameters().LE_;
    
    // Components of k-vector
    double kx = ( k(0) * b1_(0) + k(1) * b2_(0) ) / W,
           ky = ( k(0) * b1_(1) + k(1) * b2_(1) ) / Le;

    // Nearest-neighbor hopping phase factors
    std::complex<double> gamma1 = exp( I * ( kx * d1_(0) + ky * d1_(1) ) );
    std::complex<double> gamma2 = exp( I * ( kx * d2_(0) + ky * d2_(1) ) );
    std::complex<double> gamma3 = exp( I * ( kx * d3_(0) + ky * d3_(1) ) );

    std::complex<double> gamma4 = exp( I * ( kx * d4_(0) + ky * d4_(1) ) );
    std::complex<double> gamma5 = exp( I * ( kx * d5_(0) + ky * d5_(1) ) );
    std::complex<double> gamma6 = exp( I * ( kx * d6_(0) + ky * d6_(1) ) );

    
    
    // Hamiltonian H_k
    Eigen::MatrixXcd v_k   = Eigen::MatrixXcd::Zero(4,8),
                     v_k_R = Eigen::MatrixXcd::Zero(4,8);


    
    v_k_R.block(0,0,4,4)  = I * d3_(0) * H_k0_R_1_ * gamma3;
    v_k_R.block(0,0,4,4) += I * d1_(0) * H_k0_R_2_ * gamma1;
    v_k_R.block(0,0,4,4) += I * d2_(0) * H_k0_R_3_ * gamma2; 

    //H_k  = H_k0_ex_;
    v_k.block(0,0,4,4)  = I * H_k0_bare_ * ( d1_(0) * gamma1 + d2_(0) * gamma2 + d3_(0) * gamma3 );
    v_k.block(0,0,4,4) += v_k_R.block(0,0,4,4);
    v_k.block(0,0,4,4) += I * H_k0_KM_   * ( d4_(0) * gamma4 + d5_(0) * gamma5 + d6_(0) * gamma6 );

    
    
    Eigen::Matrix4cd v_k_adj = v_k.block(0,0,4,4).adjoint(); 
    v_k.block(0,0,4,4) = v_k.block(0,0,4,4)  + v_k_adj; 




    
    v_k_R.block(0,4,4,4)  = I * d3_(1) * H_k0_R_1_ * gamma3;
    v_k_R.block(0,4,4,4) += I * d1_(1) * H_k0_R_2_ * gamma1;
    v_k_R.block(0,4,4,4) += I * d2_(1) * H_k0_R_3_ * gamma2; 

    //H_k  = H_k0_ex_;
    v_k.block(0,4,4,4)  = I * H_k0_bare_ * ( d1_(1) * gamma1 + d2_(1) * gamma2 + d3_(1) * gamma3 );
    v_k.block(0,4,4,4) += v_k_R.block(0,4,4,4);
    v_k.block(0,4,4,4) += I * H_k0_KM_   * ( d4_(1) * gamma4 + d5_(1) * gamma5 + d6_(1) * gamma6 );

    
    
    v_k_adj = v_k.block(0,4,4,4).adjoint(); 
    v_k.block(0,4,4,4) = v_k.block(0,4,4,4)  + v_k_adj; 
    

        

    
    return v_k;  
}




void Graphene_KaneMele::J (type* ket, type* p_ket, int dir){

  int SUBDIM = this->parameters().SUBDIM_;
  

  
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
 for(int i=0; i<SUBDIM/4; i++){
    
      eig_ket.segment(i,2)   =  chosen_dir * eig_p_ket.segment(i,2);
      eig_ket.segment(i+2,2) =  chosen_dir * eig_p_ket.segment(i+2,2);
 
    }

  /*
  int C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;

#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){      
      int n1 = ( ( j + C ) * W + i ) * 4;

      eig_ket.segment(n1,2)   =  chosen_dir * eig_p_ket.segment(n1,2);
      eig_ket.segment(n1+2,2) =  chosen_dir * eig_p_ket.segment(n1+2,2);
 
    }
    }*/
 
};


void Graphene_KaneMele::print_hamiltonian(){
  size_t dim = this->parameters().DIM_;
  size_t DIM = this->parameters().DIM_;

  Eigen::MatrixXcd H_r(dim,dim), H_r_2(dim,dim), S(dim,dim);

  std::ofstream dataP;
  dataP.open("vx_diff.txt");

  
    for(size_t j=0;j<dim;j++){
      for(size_t i=0;i<dim;i++){

	Eigen::Vector<std::complex<double>, -1>
	  term_i=Eigen::Vector<std::complex<double>, -1>::Zero(DIM),
	  tmp=term_i,
	  term_j=term_i,
	  null=Eigen::Vector<std::complex<double>, -1>::Zero(DIM);

	Eigen::Matrix<std::complex<double>, -1, 1>  null2=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM);
	term_i(i)=1;
	term_j(j)=1;


	Eigen::Vector<std::complex<double>, -1> term_i_2=term_i, term_j_2=term_j;



	
	
	//this->Hr_ket(1.0, 0.0, tmp.data(),term_j.data());
        //this->Hr_update_cheb(tmp.data(),term_j.data(),null.data());
	
	//vel_op_x(tmp.data(),term_j.data());
        vel_op_x(tmp.data(),term_j.data());
	std::complex<double> termy = term_i.dot(tmp);

	 H_r(i,j)=termy;


	to_kSpace_pruned_2(term_j_2.data(), term_j_2.data());
	//this->Hk_ket_cut(1.0, 0.0, tmp.data(),term_j_2.data()); //k_vel_op_x_cut(tmp.data(),term_j_2.data());//
	//this->Hk_update_cheb_cut(tmp.data(),term_j_2.data(),null2.data());
	k_vel_op_x_cut(tmp.data(),term_j_2.data());
        //k_vel_op_y_cut(tmp.data(),term_j_2.data());
	
	to_rSpace_pruned_2(tmp.data(), tmp.data());
	
	std::complex<double> termy2 = term_i_2.dot(tmp);

	std::complex<double> res =   termy - termy2 ; 
       	if(abs(real(res)) < 0.00001)
	  res = std::complex<double>(0.0, imag(res));
	if(abs(imag(res)) < 0.00001)
	  res = std::complex<double>(real(res), 0.0);
	
	//if(abs(termy2-termy)>0.0000001)
	  H_r_2(i,j)=res;
      }
    }

    dataP<<H_r_2;

    std::cout<<"IS ZERO??  "<<H_r_2.norm()<<std::endl;
    std::cout<<(H_r-H_r.adjoint()).norm()<<std::endl;
  dataP.close();

  }


void Graphene_KaneMele::traceover(type* traced, type* full_vec, int s, int num_reps){
  size_t subDim = this->parameters().SUBDIM_,
      C   = this->parameters().C_,
      W   = this->parameters().W_,
      sec_size = subDim/num_reps,
      buffer_length = sec_size;
	
  if( s == num_reps-1 )
      buffer_length += subDim % num_reps;

  
 

  

  
  if(k_space_ && this->parameters().dis_str_ !=0 && false){
#pragma omp parallel for  
  for (size_t i = s * sec_size/4; i < size_t(buffer_length)/4; i++ ) {
    const auto& vec = nonZeroList_[i]; 
    size_t i2 = vec(0);
    size_t j2 = vec(1);


    for(size_t j = 0; j < 4; j++)
      traced[ i*4 + j - s * sec_size ] = full_vec[ ( i2 + j2 * W ) * 4 + j ];

  }
}
  else{
      
#pragma omp parallel for 
      for(size_t i=0;i<buffer_length;i++)
        traced[i] = full_vec[s*sec_size + i+C*W];

  }
    
};


void Graphene_KaneMele::projector(type* ket ){

  size_t DIM = parameters().DIM_;
  Eigen::VectorXcd diag(DIM);

  int option = parameters().projector_option_;
  
  diag.setOnes();
  //option 0 is no projection
  
  if( option == 1 )//spin up
    for( size_t i = 0; i < DIM; i++ )
      if( i % 2 == 0 )
	diag(i)=0;
  

  if( option == 2 )//spin down
    for( size_t i = 0; i < DIM; i++ )
      if( (i+1) % 2 == 0 )
	diag(i)=0;


  if( option == 3 )//A sublattice
    for( size_t i = 0; i < DIM/2; i++ )
      if( i % 2 == 0 ){
	diag(i) = 0;
	diag(i+1) = 0;	
      }


  if( option == 4 )//B sublattice
    for( size_t i = 0; i < DIM/2; i++ )
      if( (i+1) % 2 == 0 ){
	diag(i) = 0;
	diag(i+1) = 0;	
      }


  Eigen::SparseMatrix<type> sparse_diag_matrix(DIM,DIM);

  sparse_diag_matrix.reserve(Eigen::VectorXi::Constant(DIM, 1));

  for(size_t i = 0; i < diag.size(); ++i) 
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



void Graphene_KaneMele::to_kSpace(type ket[], const type p_ket[], int dir) {
  
  size_t num_subvectors = 4;
  size_t W = this->parameters().W_,
      LE = this->parameters().LE_;
  
  double norm = std::sqrt( double(W) * double(LE));

  for (size_t i = 0; i < num_subvectors; i++) {       
  
#pragma omp parallel for
      for (size_t y = 0; y < LE; y++) {
          for (size_t x = 0; x < W; x++) {
	    if( ( i == 2 || i == 3 ) && dir == 1 ){
	      double ky = ( x * b1_(1) + y * b2_(1) ) / double(LE);
	      type phase = std::exp(std::complex<double>(0, -a0_ * ky ) );

	      fft_input_[y * W + x][0] = ( p_ket[ ( y * W + x ) * num_subvectors + i] * phase ).real(); 
              fft_input_[y * W + x][1] = ( p_ket[ ( y * W + x ) * num_subvectors + i] * phase).imag(); 
	    }
	    else{
	      fft_input_[y * W + x][0] = p_ket[ ( y * W + x ) * num_subvectors + i ].real(); 
              fft_input_[y * W + x][1] = p_ket[ ( y * W + x ) * num_subvectors + i ].imag(); 
	    }
	 }
      }
      
      if( dir == 1 )
         fftw_execute(fftw_plan_BACK_); 
      if( dir == -1 )
         fftw_execute(fftw_plan_FORWD_);
      
#pragma omp parallel for
    for (size_t y = 0; y < LE; y++) 
      for (size_t x = 0; x < W ; x++) {
	std::complex<double> res = std::complex<double> (fft_output_[y * W + x][0], fft_output_[y * W + x][1]);
	if( ( i == 2 || i == 3 ) && dir == -1 ){
	  double ky = ( y * b1_(1) + x * b2_(1) ) / double(LE);
	  type phase = std::exp(std::complex<double>(0, -a0_ * ky ) );
	  
	  res *= conj(phase);
	}
	
	ket[ ( y * W + x ) * num_subvectors + i] =  res / norm;
      }
  }
}


void Graphene_KaneMele::to_rSpace_pruned_2(type ket[], const type p_ket[]) {
  
  size_t num_subvectors = 4;
  size_t W = this->parameters().W_,
    LE = this->parameters().LE_,
    SUBDIM = this->parameters().SUBDIM_;
  
  double norm = std::sqrt( double(W) * double(LE));

  
  for (size_t i = 0; i < num_subvectors; i++) {       
  
#pragma omp parallel for
    for(size_t i=0; i<W*LE;i++){//Its not clear to me if I have to reinitilize input all times.
      fft_input_[i][0] = 0.0;
      fft_input_[i][1] = 0.0;
    }


  
#pragma omp parallel for
    for (size_t ks = 0; ks < SUBDIM/4; ks++ ) {
      const auto& vec = nonZeroList_[ks]; 
      size_t kx = vec(0);
      size_t ky = vec(1);

      if( ( i == 2 || i == 3 ) ){
	double k_phase = ( kx * b1_(1) + ky * b2_(1) ) / double(LE);
	type phase = std::exp(std::complex<double>(0, -a0_ * k_phase ) );


	fft_input_[ky * W + kx][0] = ( p_ket[ ks * num_subvectors + i] * phase ).real(); 
	fft_input_[ky * W + kx][1] = ( p_ket[ ks * num_subvectors + i] * phase ).imag(); 
      }
      else{
	fft_input_[ky * W + kx][0] = p_ket[ ks * num_subvectors + i ].real(); 
	fft_input_[ky * W + kx][1] = p_ket[ ks * num_subvectors + i ].imag(); 
      }
    }

    


    fftw_execute(fftw_plan_BACK_);


      
#pragma omp parallel for
    for (size_t y = 0; y < LE; y++) 
      for (size_t x = 0; x < W ; x++) {
	std::complex<double> res = std::complex<double> (fft_output_[y * W + x][0], fft_output_[y * W + x][1]);	
	ket[ ( y * W + x ) * num_subvectors + i] =  res / norm;
      }
  }
}




void Graphene_KaneMele::to_kSpace_pruned_2(type ket[], const type p_ket[]) {
  
  size_t num_subvectors = 4;
  size_t W = this->parameters().W_,
    LE = this->parameters().LE_,
    SUBDIM = this->parameters().SUBDIM_;
  
  double norm = std::sqrt( double(W) * double(LE));


  for (size_t i = 0; i < num_subvectors; i++) {       

#pragma omp parallel for
    for (size_t y = 0; y < LE; y++) {
          for (size_t x = 0; x < W; x++) {
	      fft_input_[y * W + x][0] = p_ket[ ( y * W + x ) * num_subvectors + i ].real(); 
              fft_input_[y * W + x][1] = p_ket[ ( y * W + x ) * num_subvectors + i ].imag(); 
	  }
    }

    fftw_execute(fftw_plan_FORWD_);    
    


      
      
#pragma omp parallel for
    for (size_t ks = 0; ks < SUBDIM/4; ks++ ) {
      const auto& vec = nonZeroList_[ks]; 
      size_t kx = vec(0);
      size_t ky = vec(1);

      std::complex<double> res = std::complex<double> (fft_output_[ky * W + kx][0], fft_output_[ky * W + kx][1]);
      if( ( i == 2 || i == 3 )  ){
	  double k_phase = ( ky * b1_(1) + kx * b2_(1) ) / double(LE);
	  type phase = std::exp(std::complex<double>(0, -a0_ * k_phase ) );

	  res *= conj(phase);

      }
      ket[ ks * num_subvectors + i] =  res / norm;
    }
  }
}







void Graphene_KaneMele::to_rSpace_pruned(type ket[], const type p_ket[]) {
  //ket is DIM and pket is SUBDIM
  
  size_t num_subvectors = 4;
  size_t W = this->parameters().W_,
    LE = this->parameters().LE_,
    SUBDIM = this->parameters().SUBDIM_;
  
  type im_j = type(0.0,1.0);



  for (size_t i = 0; i < num_subvectors; i++) {       
  
#pragma omp parallel for
      for (size_t y = 0; y < LE; y++) {
          for (size_t x = 0; x < W; x++) {
	    if( ( i == 2 || i == 3 )  ){
	      double k_phase = ( x * b1_(1) + y * b2_(1) ) / double(LE);
	      type phase = std::exp(std::complex<double>(0, -a0_ * k_phase ) );

	      ket[y * W + x] = 0;

              Eigen::Vector2d R = x * A1_ + y * A2_; 

	      for (size_t ks = 0; i < SUBDIM; ks++ ) {
		const auto& vec = nonZeroList_[ks]; 
		size_t kx = vec(0);
		size_t ky = vec(1);
		Eigen::Vector2d k = ( kx * b1_ + ky * b2_ ) / double(LE);
		ket[y * W + x] +=  std::exp(- im_j * R.dot(k) ) * phase * p_ket[ky * W + kx];
	      }	      
	    }
	    else{

	      ket[y * W + x] = 0;
              Eigen::Vector2d R = x * A1_ + y * A2_; 

	      for (size_t ks = 0; i < SUBDIM; ks++ ) {
		const auto& vec = nonZeroList_[ks]; 
		size_t kx = vec(0);
		size_t ky = vec(1);
		Eigen::Vector2d k = ( kx * b1_ + ky * b2_ ) / double(LE);
		ket[y * W + x] +=  std::exp(- im_j * R.dot(k) ) * p_ket[ky * W + kx];
	      }	      
	    }
	  }
      }
  }
}


void Graphene_KaneMele::to_kSpace_pruned(type ket[], const type p_ket[]) {
  //ket is SUBDIM and pket is DIM
  
  size_t num_subvectors = 4;
  size_t W = this->parameters().W_,
    LE = this->parameters().LE_,
    SUBDIM = this->parameters().SUBDIM_;
  
  type im_j = type(0.0,1.0);


  for (size_t i = 0; i < num_subvectors; i++) {       
  
#pragma omp parallel for
 for (size_t ks = 0; ks < SUBDIM; ks++ ) {
    const auto& vec = nonZeroList_[i]; 
    size_t kx = vec(0);
    size_t ky = vec(1);
    Eigen::Vector2d k = ( kx * b1_ + ky * b2_ ) / double(LE);
	

    for (size_t y = 0; y < LE; y++) {
          for (size_t x = 0; x < W; x++) {
	      ket[y * W + x] = 0;
              Eigen::Vector2d R = x * A1_ + y * A2_; 

	    if( ( i == 2 || i == 3 )  ){
	      double k_phase = ( y * b1_(1) + x * b2_(1) ) / double(LE); //The transpose
	      type phase = conj(std::exp(std::complex<double>(0, -a0_ * k_phase ) )); //The conj

	      ket[ks] +=  std::exp( im_j * R.dot(k) ) * phase * p_ket[y * W + x ];
	    }	      
	  
	    else
	      ket[ks] +=  std::exp( im_j * R.dot(k) ) * p_ket[ky * W + kx];
	      	      
	    }
	  }
      }
  }
}




void Graphene_KaneMele::Hr_ket (r_type a, r_type b, type* ket, type* p_ket){

  size_t Le = this->parameters().LE_,
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
 for(size_t j=0; j<Le; j++){
    for(size_t i=0; i<W; i++){      
      size_t n1 = ( j * W + i ) * 4;

      
      eig_ket.segment(n1,4) =  H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      int n2 = (  j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) +=  H_2  * eig_p_ket.segment(n2,4);
      
      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3  * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j + 1 ) % Le ) * W    +    ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_4 * eig_p_ket.segment(n2,4);
      

      
      //Adjoints      
      n2 = ( j * W + ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);
      
      n2 = (  ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W +   ( i + 1 ) % W  ) * 4;
      eig_ket.segment(n1,4) += H_4.adjoint() * eig_p_ket.segment(n2,4);
      

      eig_ket.segment(n1,4) *= dmp_op[n1/4];
    }
 }

 if( disorder_potential != NULL ){
  #pragma omp parallel for
 for(size_t i = 0; i < this->parameters().DIM_/2; i++){
    eig_ket[2*i] += disorder_potential[i] * eig_p_ket[2*i] / a;
    eig_ket[2*i+1] += disorder_potential[i] * eig_p_ket[2*i+1] / a;
  }
}
  /*
 if( disorder_potential != NULL )
   for( size_t i = 0; i < 2 * W * Le ; i ++ )
     eig_ket.segment(i,2) += dmp_op[i/2] * disorder_potential[ i ] * eig_p_ket.segment(i,2) / a;
  */
};



void Graphene_KaneMele::Hr_update_cheb ( type ket[], type p_ket[], type pp_ket[]){
  size_t Le = this->parameters().LE_,
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
 for(size_t j=0; j<Le; j++){
    for(size_t i=0; i<W; i++){      
      size_t n1 = ( j * W + i ) * 4;
      
      eig_ket.segment(n1,4) = b_a  * eig_p_ket.segment(n1,4) - 0.5 * dmp_op[n1/4] * eig_pp_ket.segment(n1,4);

      
      eig_ket.segment(n1,4) += H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      
      size_t n2 = (  j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) +=  H_2  * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3  * eig_p_ket.segment(n2, 4);

      n2 = ( ( ( j + 1 ) % Le ) * W    +    ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_4 * eig_p_ket.segment(n2,4);
      



      //Adjoints
      n2 = ( j * W + ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);

      n2 = (  ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W  +   ( i + 1 ) % W  ) * 4;
      eig_ket.segment(n1,4) += H_4.adjoint() * eig_p_ket.segment(n2,4);


      
      eig_ket.segment(n1,4) *= 2.0 * dmp_op[n1/4];
      
      eig_pp_ket.segment(n1,4) = eig_p_ket.segment(n1,4); 
    }
 }

 if( disorder_potential != NULL ){
  #pragma omp parallel for
 for(size_t i = 0; i < this->parameters().DIM_/2; i++){
    eig_ket[2*i] += 2.0 * disorder_potential[i] * eig_p_ket[2*i] / a;
    eig_ket[2*i+1] += 2.0 * disorder_potential[i] * eig_p_ket[2*i+1] / a;
  }
 }
 /*
 if( disorder_potential != NULL )
   for( size_t i = 0; i < 2 * W * Le ; i ++ )
     eig_ket.segment(i,2) += 2.0 * dmp_op[i] * disorder_potential[ i ] * eig_p_ket.segment(i,2) / a;
 */
 
 eig_p_ket = eig_ket;
 
};




void Graphene_KaneMele::update_cheb_filtered ( type ket[], type p_ket[], type pp_ket[], type disp_factor){
  size_t Le = this->parameters().LE_,
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
 for(size_t j=0; j<Le; j++){
    for(size_t i=0; i<W; i++){      
      size_t n1 = ( j * W + i ) * 4;
      
      eig_ket.segment(n1,4) = b_a  * eig_p_ket.segment(n1,4) - 0.5 * disp_factor * dmp_op[n1/4] * eig_pp_ket.segment(n1,4);

      
      eig_ket.segment(n1,4) += H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      
      size_t n2 = (  j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) +=  H_2  * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3  * eig_p_ket.segment(n2, 4);

      n2 = ( ( ( j + 1 ) % Le ) * W    +       ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4) += H_4 * eig_p_ket.segment(n2,4);
      



      //Adjoints
      n2 = ( j * W + ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_2.adjoint() * eig_p_ket.segment(n2,4);

      n2 = (  ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W + i ) * 4;
      eig_ket.segment(n1, 4) += H_3.adjoint() * eig_p_ket.segment(n2, 4);
      
      n2 = ( ( ( j - 1 ) == -1 ? ( Le - 1 ) : ( j - 1 ) ) * W +  ( ( i - 1 ) == -1 ? ( W - 1 ) : ( i - 1 ) )  ) * 4;
      eig_ket.segment(n1,4) += H_4.adjoint() * eig_p_ket.segment(n2,4);


      
      eig_ket.segment(n1,4) *= 2.0 * disp_factor * dmp_op[n1/4];
      
      eig_pp_ket.segment(n1,4) = eig_p_ket.segment(n1,4); 
    }
 }


 
 if( disorder_potential != NULL )
   for( size_t i = 0; i < 2 * W * Le ; i ++ )
     eig_ket.segment(i,2) += 2.0 * dmp_op[i] * disorder_potential[ i ] * eig_p_ket.segment(i,2)/a;

 
 eig_p_ket = eig_ket;
 
};





void Graphene_KaneMele::vel_op_y (type* ket, type* p_ket){

  size_t Le = this->parameters().LE_,
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
    d_y  = j1  * a0_, 
    d_y2 = j1 * sin( M_PI / 6.0 )  * a0_,
    d_y3 = j1 * 3.0 / 2.0  * a0_;


  
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
 for(size_t j=0; j<Le; j++){
    for(size_t i=0; i<W; i++){      
      size_t n1 = ( (j) * W + i ) * 4;

      
      eig_ket.segment(n1,4) = H_1 * eig_p_ket.segment(n1,4);
      eig_ket.segment(n1,4) += H_1.adjoint() * eig_p_ket.segment(n1,4);

      
      size_t n2 = ( ( j ) * W + ( i + 1 ) % W ) * 4;
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

  size_t Le = this->parameters().LE_,
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
    d_x  = -j1 * cos( M_PI /6.0 ) * a0_,
    d_x2 = -j1 * 0.5 * sqrt( 3.0 )  * a0_;


    
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
 for(size_t j=0; j<Le; j++){
    for(size_t i=0; i<W; i++){      
      size_t n1 = ( j * W + i ) * 4;

      

      size_t n2 = ( j * W + ( i + 1 ) % W ) * 4;
      eig_ket.segment(n1,4)   =  H_2 * eig_p_ket.segment(n2,4);

      n2 = ( ( ( j + 1 ) % Le ) * W + i ) * 4;
      eig_ket.segment(n1, 4) +=  H_3 * eig_p_ket.segment(n2, 4);

      n2 = ( ( ( j + 1 ) % Le )* W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      eig_ket.segment(n1,4)  +=  H_4 * eig_p_ket.segment(n2,4);

      

      //Adjoints
      n2 = ( ( j ) * W + ( ( i - 1 ) == -1 ? (W-1) : (i-1) )  ) * 4;
      eig_ket.segment( n1, 4 ) +=   H_2.adjoint() * eig_p_ket.segment(n2,4);
      
      n2 = (  ( ( j - 1 ) == -1 ? (Le - 1) : (j-1) ) * W + i ) * 4;
      eig_ket.segment( n1, 4 ) +=  H_3.adjoint() * eig_p_ket.segment(n2, 4);
	
      n2 = ( ( ( j - 1 ) == -1 ? (Le-1) : (j-1) ) * W  + ( i + 1 ) % W   ) * 4;
      eig_ket.segment( n1, 4 ) +=  H_4.adjoint() * eig_p_ket.segment(n2,4);

    }
 }

 //eig_ket *= sqrt(3)/3;
 
};





 




