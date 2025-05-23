#ifndef VIET_MATBG_HPP
#define VIET_MATBG_HPP


#include "Device.hpp"
#include <eigen3/Eigen/Sparse>
#include "Coordinates.hpp"


class VietMATBG: public Device{
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor> SpMatrixXp;
  typedef Eigen::Matrix<r_type,-1,-1> MatrixXp;
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;

private:
  bool print_CSR = true;
  SpMatrixXp H_, vx_; 
  Coordinates coordinates_,
              moire_unitCell_coordinates_;

  r_type a_ = 1.0,
         b_ = 0.0;

  
public:
  ~VietMATBG(){};
  VietMATBG(device_vars& );

  
  virtual r_type Hamiltonian_size(){return H_.rows();};  

  virtual void build_Hamiltonian();
  virtual void setup_velOp() ;
  virtual void adimensionalize ( r_type a,  r_type b){    
      int Dim = this->parameters().DIM_;
      SpMatrixXp Id(Dim,Dim);
      Id.setIdentity();

      H_=(H_+b*Id)/a;
  };
  virtual void damp   ( r_type*) ;



  virtual void update_cheb ( type*, type*, type*, r_type*, r_type* ) ;  
  virtual void H_ket ( type*, type*, r_type*, r_type*) ;
  virtual void vel_op (type*, type*) ;




  
  //Unused for now  
  virtual void update_dis(r_type*,r_type*) {};//Disorder will not be supported for now; Tis` trivial to support it though
  
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type , r_type )  {};
  virtual void update_cheb_filtered ( type*, type*, type*, r_type*, r_type* , type) {};  
  virtual void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type )  {};  
  virtual void H_ket ( type*, type*) {};
  

};

#endif //READ_HAMILTONIAN_HPP
