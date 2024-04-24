#ifndef READ_HAMILTONIAN_HPP
#define READ_HAMILTONIAN_HPP


#include "Device.hpp"
#include <eigen-3.4.0/Eigen/Sparse>
#include "Coordinates.hpp"


class Read_Hamiltonian: public Device{
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor> SpMatrixXp;
  typedef Eigen::Matrix<r_type,-1,-1> MatrixXp;
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;

private:
  bool print_CSR = true;
  SpMatrixXp H_, vx_; 
  Coordinates coordinates_;

  r_type a_ = 1.0,
         b_ = 0.0;

  
public:
  ~Read_Hamiltonian(){};
  Read_Hamiltonian(device_vars& );

  
  virtual r_type Hamiltonian_size(){return ( H_.innerSize() + H_.nonZeros() + H_.outerSize() ) * sizeof(r_type);};  

  virtual void build_Hamiltonian();
  virtual void setup_velOp() ;
  virtual void adimensionalize ( r_type a,  r_type b){

      a_=a, b_=b;
      int Dim = this->parameters().DIM_;
      SpMatrixXp Id(Dim,Dim);
      Id.setIdentity();

      H_=(H_+b*Id)/a;
      vx_=vx_/a;

  };
  virtual void damp   ( r_type*) ;



  virtual void update_cheb ( type*, type*, type*, r_type*, r_type* ) ;  
  virtual void H_ket ( type*, type*, r_type*, r_type*) ;
  virtual void vel_op (type*, type*) ;




  
  //Unused for now  
  virtual void update_dis(r_type*,r_type*);//Disorder will not be supported for now; Tis` trivial to support it though
  
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type , r_type )  {};
  virtual void update_cheb_filtered ( type*, type*, type*, r_type*, r_type* , type) {};  
  virtual void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type )  {};  
  virtual void H_ket ( type*, type*) {};
  

};

#endif //READ_HAMILTONIAN_HPP
