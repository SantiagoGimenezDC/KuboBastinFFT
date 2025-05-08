#ifndef READ_HAMILTONIAN_HPP
#define READ_HAMILTONIAN_HPP


#include "Device.hpp"
#include <eigen3/Eigen/Sparse>
#include "Coordinates.hpp"


class Read_Hamiltonian: public Device{
  typedef long int indexType;
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor, indexType> SpMatrixXp;
  typedef Eigen::SparseMatrix<type,Eigen::RowMajor, indexType> SpMatrixXcp;
  typedef Eigen::Matrix<r_type,-1,-1> MatrixXp;
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;

private:
  bool print_CSR = true;
  SpMatrixXp H_, vx_;
  SpMatrixXcp Hc_, vxc_; 
  Coordinates coordinates_;

  r_type a_ = 1.0,
         b_ = 0.0;

  
public:
  ~Read_Hamiltonian(){};
  Read_Hamiltonian(device_vars& );

  Coordinates& coordinates(){return coordinates_;};
  void set_coordinates(Coordinates new_coordinates){coordinates_ = new_coordinates;};
  void set_H(Eigen::Map<Eigen::SparseMatrix<type, Eigen::RowMajor >> & new_H){ Hc_ = new_H; };
  SpMatrixXcp& H(int){return Hc_;};
  SpMatrixXcp& vx(int){return vxc_;};

  void set_H(Eigen::Map<Eigen::SparseMatrix<r_type, Eigen::RowMajor >> & new_H){ H_ = new_H; };
  SpMatrixXp& H(){return H_;};
  SpMatrixXp& vx(){return vx_;};

  
  virtual r_type Hamiltonian_size(){
    if(H_.size()>0)
      return ( H_.innerSize() + H_.nonZeros() + H_.outerSize() ) * sizeof(r_type);
    if(Hc_.size()>0)
      return ( Hc_.innerSize() + Hc_.nonZeros() + Hc_.outerSize() ) * sizeof(type);
  };  




  virtual void build_Hamiltonian();
  virtual void setup_velOp() ;
  virtual void adimensionalize ( r_type a,  r_type b){

      a_=a, b_=b;
      int Dim = this->parameters().DIM_;
      SpMatrixXp Id(Dim,Dim);
      SpMatrixXcp Id2(Dim,Dim);
      Id.setIdentity();
      Id2.setIdentity();

     if(H_.size()>0){
      H_=(H_+b*Id)/a;
      vx_=vx_/a;
      }
     if(Hc_.size()>0){
      Hc_=(Hc_+b*Id2)/a;
      vxc_=vxc_/a;
     }
  };
  virtual void damp   ( r_type*) ;


  virtual void update_cheb ( type*, type*, type*) ;  
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type* ) ;  

  virtual void H_ket ( type*, type*) ;
  virtual void H_ket ( type*, type*, r_type*, r_type*) ;
  virtual void vel_op (type*, type*) ;
  virtual void vel_op (type* vec, type* p_vec, int){vel_op (vec, p_vec);};




  
  //Unused for now  
  virtual void update_dis(r_type*,r_type*);//Disorder will not be supported for now; Tis` trivial to support it though
  
  virtual void update_cheb ( type*, type*, type*, r_type*, r_type , r_type )  {};
  virtual void update_cheb_filtered ( type*, type*, type*, r_type*, r_type* , type) {};  
  virtual void update_cheb ( int ,  int, type*, type*, type*, type*, r_type*, r_type , r_type )  {};  
  
  

};

#endif //READ_HAMILTONIAN_HPP
