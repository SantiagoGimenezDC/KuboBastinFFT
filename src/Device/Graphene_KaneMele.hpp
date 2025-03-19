#ifndef GRAPHENE_KANEMELE_H
#define GRAPHENE_KANEMELE_H

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include "Device.hpp"
#include "Graphene.hpp"
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include "Coordinates.hpp"
#include <iostream>




struct eigenSol {
    Eigen::Vector4cd eigenvalues_;  // Complex vector with 4 entries
    Eigen::Matrix4cd Uk_;  // 4x4 complex matrix
};



class Graphene_KaneMele: public Graphene{

  typedef Eigen::SparseMatrix<std::complex<r_type>,Eigen::RowMajor> SpMatrixXpc;
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor> SpMatrixXp;
  typedef Eigen::Matrix<r_type,-1,-1> MatrixXp;
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;
    
  private:

    bool CYCLIC_BCs_ = false;
    r_type peierls_d_=0;

    const r_type e_standard_ = 0.0;
    const r_type t_standard_ = - 2.7;
    const r_type  a0_        = 0.142;
    r_type stgr_str_         = 0.0;
    r_type m_str_            = 0.0;
    r_type rashba_str_       = 0.0;  
    r_type KM_str_           = 0.0;  

    Eigen::Matrix2cd sx{{0,1},{1,0}}, sy{{0,-type(0,1)}, {type(0,1), 0}}, sz{{1,0}, {0, -1}};


    Eigen::Matrix4cd
      H_HLD_ = Eigen::Matrix4d::Zero(),
      H_KM_ = Eigen::Matrix4d::Zero(),
      H_1_ = Eigen::Matrix4d::Zero(),
      H_2_ = Eigen::Matrix4d::Zero(),
      H_3_ = Eigen::Matrix4d::Zero(),
      H_4_ = Eigen::Matrix4d::Zero();



  
    double sqrt3_ = std::sqrt(3.0);
    double a_ = sqrt3_ * a0_;

  
    Eigen::Vector2d A1 = a_ * Eigen::Vector2d(1/2, sqrt3_/2);
    Eigen::Vector2d A2 = a_ * Eigen::Vector2d(-1/2, sqrt3_/2);

  
    Eigen::Vector2d b1_ = ( 2.0 * M_PI / (sqrt3_ * a_)) * Eigen::Vector2d( sqrt3_, 1);
    Eigen::Vector2d b2_ = ( 2.0 * M_PI / (sqrt3_ * a_)) * Eigen::Vector2d(-sqrt3_, 1);


  
      // Nearest-neighbor vectors
    Eigen::Vector2d d1_ = a0_ * Eigen::Vector2d(sqrt3_ / 2, 1.0 / 2);
    Eigen::Vector2d d2_ = a0_ * Eigen::Vector2d(-sqrt3_ / 2, 1.0 / 2);
    Eigen::Vector2d d3_ = a0_ * Eigen::Vector2d(0, -1);
    
    Eigen::Vector2d d4_ = Eigen::Vector2d(0.5 * a_, sqrt3_ / 2 * a_);
    Eigen::Vector2d d5_ = Eigen::Vector2d(0.5 * a_, -sqrt3_ / 2 * a_);
    Eigen::Vector2d d6_ = Eigen::Vector2d(-a_, 0);


    Eigen::Matrix4cd H_k0_R_1_ =  Eigen::Matrix4d::Zero(),
                     H_k0_R_2_ = Eigen::Matrix4d::Zero(),
                     H_k0_R_3_ = Eigen::Matrix4d::Zero();

    
    Eigen::Matrix4cd H_k0_ex_ = Eigen::Matrix4d::Zero(),
                     H_k0_bare_ = Eigen::Matrix4cd::Zero(),
                     H_k0_KM_ = Eigen::Matrix4cd::Zero(),
                     H_k0_R_ = Eigen::Matrix4cd::Zero();

    Eigen::MatrixXcd phases_;

    std::vector<eigenSol> diagonalized_Hk_;
    Eigen::VectorXcd eigenvalues_k_;
    Eigen::MatrixXcd H_k_, U_k_, v_k_x_, v_k_y_, v_k_z_;

    bool k_space = false;

  
  public:
    ~Graphene_KaneMele(){};
    Graphene_KaneMele();
    Graphene_KaneMele(r_type, r_type, r_type , r_type, r_type, device_vars&);

    void print_hamiltonian();
    virtual void rearrange_initial_vec(type*){};
    virtual void traceover(type* , type* , int , int);

    virtual bool isKspace(){ return k_space; };  
    virtual int unit_cell_size(){return 4;};  
    virtual void projector(type* );  
    virtual void J (type*, type*, int);

    virtual void build_Hamiltonian(){};

    //Rashba coupling Hamiltonian //INTERFACE
    virtual void H_ket  ( type* ket , type* p_ket ){
      if(k_space)
	Hk_ket(this->a(),this->b(), ket, p_ket);
      else
        Hr_ket(this->a(),this->b(), ket, p_ket);

    }
  
    virtual void H_ket  ( type* ket , type* p_ket, r_type*, r_type* ){
      if(k_space)
        Hk_ket(this->a(), this->b(), ket, p_ket);
      else
	Hr_ket(this->a(), this->b(), ket, p_ket);
    }
  
    virtual void update_cheb ( type* ket, type* p_ket,  type* pp_ket){
      if(k_space)
        Hk_update_cheb( ket, p_ket, pp_ket);
      else
	Hr_update_cheb( ket, p_ket, pp_ket);
    };




  //Real space
    
    void Hr_ket  ( r_type, r_type, type*, type* );
    void Hr_update_cheb ( type*, type*,  type* );
  
    virtual void update_cheb_filtered ( type ket[], type p_ket[], type pp_ket[], r_type*, r_type*, type disp_factor){
      update_cheb_filtered ( ket, p_ket, pp_ket, disp_factor );
    };
    void update_cheb_filtered ( type *, type *, type *,  type );



  
    //K space
    virtual void to_kSpace(type* , const type*, int );
    void diagonalize_kSpace();
    eigenSol  Uk_single(Eigen::Vector2d );
  
    Eigen::Matrix4cd Hk_single(Eigen::Vector2d );
    Eigen::MatrixXcd vk_single(Eigen::Vector2d );
  
    void Hk_ket ( r_type, r_type, type*, type* );
    virtual void Uk_ket (  type*, type* );
    void Hk_update_cheb ( type*, type*,  type*);

    void k_vel_op_x (type*, type* );
    void k_vel_op_y (type*, type* );  



  
    virtual void vel_op   ( type* ket, type* p_ket, int dir){
      if( dir == 0 ){
	if(k_space)
	  k_vel_op_x( ket, p_ket);
	else
	  vel_op_x( ket, p_ket);
      }
      if( dir == 1 ){
	if(k_space)
	  k_vel_op_y( ket, p_ket);
	else
          vel_op_y( ket, p_ket);
      }

      
      
      if( dir == 2 )
        this->J( ket, p_ket, 0);
      if( dir == 3 )
        this->J( ket, p_ket, 1);
      if( dir == 4 )
        this->J( ket, p_ket, 2);
    
    };
  
    virtual void vel_op_x   ( type*, type*);
    virtual void vel_op_y ( type*, type*);  
};






#endif // ARMCHAIR_GRAPHENE_H
