#include<iostream>
#include<fstream>
#include<chrono>
#include "VietMATBG.hpp"
#include<fstream>


VietMATBG::VietMATBG(device_vars& device_vars):Device(device_vars){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  inFile.open("VietMATBG.moireCell.dat");


    
    
  std::size_t DIM, dump, UC00, UC01, UC10, UC11;

  inFile>>UC00;
  inFile>>UC01;
  inFile>>UC10;
  inFile>>UC11;
  inFile>>dump;
  inFile>>dump;
  inFile>>DIM;  

  MatrixXp coordinates(DIM, 3);


  
  for(std::size_t i=0; i<DIM; i++)
    for(int j=0; j<3;j++)
      inFile>>coordinates(i, j); 

  
  moire_unitCell_coordinates_.reset(coordinates);

};


void VietMATBG::build_Hamiltonian(){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  inFile.open(run_dir+"operators/"+filename+".CSR");

  std::cout<<"  Imaginary part of the Hamiltonian is being dumped on read;"<<std::endl<<std::endl;
    
    
  std::size_t DIM, NNZ;
  inFile>>DIM;
  inFile>>NNZ;
  
  
  parameters().DIM_    = DIM;
  parameters().SUBDIM_ = DIM;  
  parameters().C_      = 0;
  parameters().W_      = 1;
  parameters().LE_     = DIM;


  int outerIndexPtr[DIM+1];
  int innerIndices[NNZ];
  r_type values[NNZ];


  for(std::size_t j=0; j<NNZ; j++){
    double re_part , im_part;
    inFile>>re_part, inFile>>im_part;
    values[j] = re_part;// + std::complex<r_type>(0,1) * type( im_part );
  }

  for(std::size_t j=0; j<NNZ; j++)  
    inFile>>innerIndices[j];

  for(std::size_t j=0; j<DIM+1; j++)  
    inFile>>outerIndexPtr[j];    

  
  Eigen::Map<Eigen::SparseMatrix<r_type> > sm1(DIM,DIM,NNZ,outerIndexPtr, // read-write
                               innerIndices,values);

  
  H_=sm1;


};



void VietMATBG::vel_op (type vec[], type p_vec[]){
  int Dim = this->parameters().DIM_;
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);

  eig_vec = vx_ * eig_p_vec;
  
};


void VietMATBG::H_ket ( type* vec, type* p_vec, r_type* dmp_op, r_type* dis_vec) {
  int Dim = this->parameters().DIM_,
      W = this->parameters().W_,
      C = this->parameters().C_,
      Le = this->parameters().LE_;
  
  
#pragma omp parallel for
  for(int i = 0; i < Dim; i++)
    p_vec[ i ] *= dmp_op[ i ];
  

  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);
  

  
  eig_vec = H_ * eig_p_vec;   

    
#pragma omp parallel for
  for(int i = 0; i < Le*W; i++) 
    vec[ i + C * W ]    +=  dis_vec[ i ] * p_vec[ i + C * W ]/a_;
  
}


void VietMATBG::update_cheb ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type*){

  int Dim = this->parameters().DIM_;
 
#pragma omp parallel for
  for(int i = 0; i < Dim; i++)
    pp_vec[ i ] *= damp_op[ i ] * damp_op[ i ];

    
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim),
    eig_pp_vec(pp_vec, Dim);
  

  
  eig_vec = 2.0 * H_ * eig_p_vec - eig_pp_vec;

    
  eig_pp_vec = eig_p_vec;
  eig_p_vec = eig_vec;
}



void VietMATBG::damp ( r_type damp_op[]){
  
  int Dim = this->parameters().DIM_;
 
  SpMatrixXp Id(Dim,Dim), gamma(Dim,Dim);//, dis(Dim,Dim);  dis.setZero();
  Id.setIdentity();
  gamma = Id;


  #pragma omp parallel for
  for(int i=0; i<Dim;i++)
    gamma.coeffRef(i,i) *=damp_op[ i ];

  H_ = gamma*H_;
}


void VietMATBG::setup_velOp(){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  inFile.open(run_dir+"operators/"+filename+"VX.CSR");

  std::cout<<"  Imaginary part of the Hamiltonian is being dumped on read;"<<std::endl<<std::endl;
    
    
  std::size_t DIM, NNZ;
  inFile>>DIM;
  inFile>>NNZ;
  

  int outerIndexPtr[DIM+1];
  int innerIndices[NNZ];
  r_type values[NNZ];


  for(std::size_t j=0; j<NNZ; j++){
    double re_part , im_part;
    inFile>>re_part, inFile>>im_part;
    values[j] = re_part;// + std::complex<r_type>(0,1) * type( im_part );
  }

  for(std::size_t j=0; j<NNZ; j++)  
    inFile>>innerIndices[j];

  for(std::size_t j=0; j<DIM+1; j++)  
    inFile>>outerIndexPtr[j];    

  
  Eigen::Map<Eigen::SparseMatrix<r_type> > sm1(DIM,DIM,NNZ,outerIndexPtr, // read-write
                               innerIndices,values);

  
  vx_=sm1;


};
