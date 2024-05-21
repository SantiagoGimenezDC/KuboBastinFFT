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
  typedef long int indexType;
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor, indexType> SpMatrixXp;
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


      
    const double V0pi_ = -2.7;//eV
    const double V0sigma_ = 0.3675;//eV
    const double qpibya0_ = 2.218;//A-1
    const double qsigmabyb0_ = qpibya0_;
    const double a0_ = 1.42;//A
    const double d0_ = 3.43;//A
    const double r0_ = 6.14;//A
    const double lambdac_ = 0.265;//A

  
public:
  ~Read_ConTable(){};
  Read_ConTable(device_vars& );
  
  virtual void build_Hamiltonian();
  virtual void setup_velOp();
  
  //  void generate_Hamiltonian();
  //void generate_velOp();    

};

#endif //READ_HAMILTONIAN_HPP
