#ifndef SUPERCELL_GRAPHENE_SOC_H
#define SUPERCELL_GRAPHENE_SOC_H

#include "../static_vars.hpp"
#include "../CAP.hpp"
#include "../Random.hpp"
#include "Device.hpp"
#include "Graphene.hpp"
#include <eigen-3.4.0/Eigen/Sparse>
#include "Coordinates.hpp"
#include <iostream>



class SupercellGraph_RashbaSOC: public Graphene{

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

    
  public:
    virtual ~SupercellGraph_RashbaSOC(){};
    SupercellGraph_RashbaSOC();
    SupercellGraph_RashbaSOC(r_type m_str, r_type rashba_str, device_vars& parameters): Graphene(parameters), m_str_(m_str), rashba_str_(rashba_str){
      this->parameters().DIM_*=2;
      this->parameters().SUBDIM_*=2;
      peierls_d_ = 2.0 * M_PI * this->parameters().Bz_ / double( 2 * (parameters.W_-1) );

      if(this->parameters().C_==0)
	CYCLIC_BCs_=true;


      if(this->parameters().W_%2!=0 || this->parameters().LE_%2==0 )
	std::cout<<"Graphene supercell oly valid for EVEN W and ODD LE!!"<<std::endl;
    };

  void print_hamiltonian();
  virtual void rearrange_initial_vec(type*);
  virtual void traceover(type* , type* , int , int);

  type peierls(int i1, int sign){
    return  std::polar(1.0, sign * ( i1 % 2 == 0 ? -1 : 1 ) * peierls_d_ * ( sign * 2 * i1 + 1  ) );
  };

  
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






#endif // ARMCHAIR_GRAPHENE_H
