#ifndef SUPERCELL_GRAPHENE_SOC_KANEMELE_H
#define SUPERCELL_GRAPHENE_SOC_KANEMELE_H

#include "../../static_vars.hpp"
#include "../../CAP.hpp"
#include "../../Random.hpp"
#include "../Device.hpp"
#include "../Graphene.hpp"
#include <eigen3/Eigen/Sparse>
#include "../Coordinates.hpp"
#include <iostream>



class SupercellGraph_Rashba_KaneMele: public Graphene{

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
    r_type m_str_            = 0.0;
    r_type rashba_str_       = 0.0;
    r_type KM_str_           = 0.0;    


    const type j1_=type(0,1);
    const Eigen::Matrix2cd sx{{0.0,1.0},{1.0,0.0}}, sy{{0.0,-j1_},{j1_,0.0}}, sz{{1.0,0.0},{0.0,-1.0}};
    
  public:
    virtual ~SupercellGraph_Rashba_KaneMele(){};
    SupercellGraph_Rashba_KaneMele();
    SupercellGraph_Rashba_KaneMele(r_type m_str, r_type rashba_str,r_type KM_str, device_vars& parameters): Graphene(parameters), m_str_(m_str), rashba_str_(rashba_str), KM_str_(KM_str){
      this->parameters().DIM_*=4;
      this->parameters().SUBDIM_*=4;


      
      if(this->parameters().C_==0)
	CYCLIC_BCs_=true;

    };

  void print_hamiltonian();
  virtual void rearrange_initial_vec(type*);
  virtual void traceover(type* , type* , int , int);


 
 
  virtual void build_Hamiltonian(){};

  //Rashba coupling Hamiltonian
  virtual void H_ket  ( type* ket , type* p_ket ){ H_ket(this->a(),this->b(), ket, p_ket); }
  virtual void H_ket  ( type* ket , type* p_ket, r_type*, r_type* ){ H_ket(this->a(),this->b(), ket, p_ket); }
  
  virtual void H_ket  (r_type, r_type,  type*, type* );
  virtual void update_cheb ( type*, type*,  type*);

  virtual void vel_op   ( type* ket, type* p_ket, int dir){
    if( dir == 0 )
      vel_op_x( ket, p_ket);
    if( dir == 1 )
      vel_op_y( ket, p_ket);
  };
  
  virtual void vel_op_x   ( type*, type*);
  virtual void vel_op_y ( type*, type*);  
};






#endif // ARMCHAIR_GRAPHENE_SOC_KANEMELEH
