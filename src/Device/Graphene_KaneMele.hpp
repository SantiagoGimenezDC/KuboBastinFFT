#ifndef GRAPHENE_KANEMELE_H
#define GRAPHENE_KANEMELE_H

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include "Device.hpp"
#include "Graphene.hpp"
#include <eigen-3.4.0/Eigen/Sparse>
#include <eigen-3.4.0/Eigen/Dense>
#include "Coordinates.hpp"
#include <iostream>



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
      H_KM_ = Eigen::Matrix4d::Zero(),
      H_1_ = Eigen::Matrix4d::Zero(),
      H_2_ = Eigen::Matrix4d::Zero(),
      H_3_ = Eigen::Matrix4d::Zero();

  
  public:
    ~Graphene_KaneMele(){};
    Graphene_KaneMele();
    Graphene_KaneMele(r_type, r_type, r_type , r_type, device_vars&);

    void print_hamiltonian();
    virtual void rearrange_initial_vec(type*){};
    virtual void traceover(type* , type* , int , int);

  
    virtual void projector(type* );  
    virtual void J (type*, type*, int);

    virtual void build_Hamiltonian(){};

    //Rashba coupling Hamiltonian
    virtual void H_ket  ( type* ket , type* p_ket ){ H_ket(this->a(),this->b(), ket, p_ket); }
    virtual void H_ket  ( type* ket , type* p_ket, r_type*, r_type* ){ H_ket(this->a(), this->b(), ket, p_ket); }
  
    virtual void H_ket  (r_type, r_type,  type*, type* );
    virtual void update_cheb ( type*, type*,  type*);

    virtual void vel_op   ( type* ket, type* p_ket, int dir){
      if( dir == 0 )
        vel_op_x( ket, p_ket);
      if( dir == 1 )
        vel_op_y( ket, p_ket);


      
      
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
