#ifndef ARMCHAIR_GRAPHENE_SOC_H
#define ARMCHAIR_GRAPHENE_SOC_H

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include "Device.hpp"
#include "Graphene.hpp"
#include <eigen-3.4.0/Eigen/Sparse>
#include "Coordinates.hpp"
#include <iostream>



class ArmchairGraph_RashbaSOC: public Graphene{

  typedef Eigen::SparseMatrix<std::complex<r_type>,Eigen::RowMajor> SpMatrixXpc;
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor> SpMatrixXp;
  typedef Eigen::Matrix<r_type,-1,-1> MatrixXp;
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;
    
  private:
  
    const r_type e_standard_ = 0.0;
    const r_type t_standard_ = -2.7;
    const r_type  a0_        = 0.142;
    r_type m_str_            = 0.0;
    r_type rashba_str_       = 0.0;  

    
  public:
    virtual ~ArmchairGraph_RashbaSOC(){};
    ArmchairGraph_RashbaSOC();
    ArmchairGraph_RashbaSOC(r_type m_str, r_type rashba_str, device_vars& parameters): Graphene(parameters), m_str_(m_str), rashba_str_(rashba_str){
      this->parameters().DIM_*=2;
      this->parameters().SUBDIM_*=2;
    };

  virtual void rearrange_initial_vec(type*);
  virtual void traceover(type* , type* , int , int);


  
  virtual void build_Hamiltonian(){};
  virtual void RashbaSOC_Hamiltonian (SpMatrixXpc&);

  //Rashba coupling Hamiltonian
  virtual void H_ket  ( type* ket , type* p_ket ){ H_ket(this->a(),this->b(), ket, p_ket); }
  virtual void H_ket  ( type* ket , type* p_ket, r_type*, r_type* ){ H_ket(this->a(),this->b(), ket, p_ket); }
  
  virtual void H_ket  (r_type, r_type,  type*, type* );
  virtual void update_cheb ( type*, type*,  type*);

  virtual void vel_op ( type*, type*);
};






#endif // ARMCHAIR_GRAPHENE_H
