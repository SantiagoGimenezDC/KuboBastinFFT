#ifndef READ_CONNTABLE_HPP
#define READ_CONNTABLE_HPP


#include "Device.hpp"
#include <eigen-3.4.0/Eigen/Sparse>
#include "Coordinates.hpp"
#include "Read_Hamiltonian.hpp"

/*The xyz file contains the following header:
U00 U01
U10 U11
a_cc
num_cutoff
num_atoms 
*/

class Read_ConTable: public Read_Hamiltonian{
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor> SpMatrixXp;
  typedef Eigen::Matrix<r_type,-1,-1> MatrixXp;
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;

private:
  Eigen::Matrix<std::size_t,-1,-1> connTable_;
  Eigen::Matrix4d U_;
  double a_cc_, num_cutoff_;
  std::size_t num_atoms_;


  
  Eigen::VectorXd vals_;
  Eigen::VectorXi  rows_;
  Eigen::VectorXi  cols_;

  
public:
  ~Read_ConTable(){};
  Read_ConTable(device_vars& );
  
  void generate_Hamiltonian();
    

};

#endif //READ_HAMILTONIAN_HPP
