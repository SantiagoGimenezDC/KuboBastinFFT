#include<iostream>
#include<fstream>
#include<chrono>
#include "Read_Hamiltonian.hpp"
#include<fstream>
#include <eigen3/Eigen/Eigenvalues>





Read_Hamiltonian::Read_Hamiltonian(device_vars& device_vars):Device(device_vars){};


void Read_Hamiltonian::build_Hamiltonian(){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  set_sysLength(1.0);  
  set_sysSubLength(1.0);
  

  inFile.precision(14);
  inFile.open(run_dir+"operators/"+filename+".HAM.CSR");

  std::cout<<run_dir+"operators/"+filename+".HAM.CSR"  <<std::endl<<std::endl;
  //"REMEMBER: vcx/hcx hack!!  Imaginary part of the Hamiltonian is being dumped on read; "<<

  std::size_t DIM, NNZ;
  inFile>>DIM;
  inFile>>NNZ;
  

 
  parameters().DIM_    = DIM;
  parameters().SUBDIM_ = DIM;  
  parameters().C_      = 0;
  parameters().W_      = 1;
  parameters().LE_     = DIM;


  
  indexType outerIndexPtr[DIM+1];
  indexType innerIndices[NNZ];
  type values[NNZ];


  for(std::size_t j=0; j<NNZ; j++){
    double re_part=0 , im_part;
     inFile>>re_part, inFile>>im_part;
     values[j] = re_part + std::complex<r_type>(0,1) * type( im_part );
  }

  for(std::size_t j=0; j<NNZ; j++)  
    inFile>>innerIndices[j];

  for(std::size_t j=0; j<DIM+1; j++)  
    inFile>>outerIndexPtr[j];    




  

  Hc_=Eigen::Map<Eigen::SparseMatrix<type, Eigen::RowMajor,indexType> > (DIM, DIM, NNZ, outerIndexPtr, innerIndices,values);

  /*  
  auto Hc_conj=Eigen::SparseMatrix<type, Eigen::RowMajor,indexType>(Hc_.transpose().conjugate());
  Hc_+=Hc_conj;
  Hc_/=2;
  int block_size= 20;
  */
  /*
  for(int i=0;i<2;i++){
    Eigen::MatrixXcd H_dense = Eigen::MatrixXcd(Hc_.block(i*block_size, i*block_size, block_size, block_size));
    //Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(H_dense);


    std::cout<<H_dense.real()<<std::endl;
    //        std::cout<<solver.eigenvalues()<<std::endl;
	
    for(int j=0;j<300;j++){
      if(solver.eigenvalues()[j].imag()!=0)

      }
  }*/
  
  inFile.close();
};



void Read_Hamiltonian::vel_op (type vec[], type p_vec[]){
  int Dim = this->parameters().DIM_;
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);


  if(vx_.size()>0)
    eig_vec = vx_ * eig_p_vec;
  if(vxc_.size()>0)
    eig_vec = vxc_ * eig_p_vec;

  
};

void Read_Hamiltonian::H_ket ( type* vec, type* p_vec ){
  H_ket(vec, p_vec, damp_op(), dis());
};


void Read_Hamiltonian::H_ket ( type* vec, type* p_vec, r_type* dmp_op, r_type* dis_vec) {
  int Dim = this->parameters().DIM_,
      subDim = this->parameters().SUBDIM_,
      W = this->parameters().W_,
      C = this->parameters().C_;
  
  /*
#pragma omp parallel for
  for(int i = 0; i < Dim; i++)
    p_vec[ i ] *= dmp_op[ i ];
  */

  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);
  
  if(H_.size()>0)
    eig_vec = H_ * eig_p_vec;
  
  else if(Hc_.size()>0)  
    eig_vec = Hc_ * eig_p_vec;

  /*
#pragma omp parallel for
  for(int i = 0; i < subDim; i++) 
    vec[ i + C * W ]    +=  dis_vec[ i ] * p_vec[ i + C * W ]/a_;
  */
}

void Read_Hamiltonian::update_cheb ( type vec[], type p_vec[], type pp_vec[]){
  update_cheb ( vec, p_vec, pp_vec, damp_op(), NULL);
};

  
void Read_Hamiltonian::update_cheb ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type*){

  int Dim = this->parameters().DIM_;
  /*
#pragma omp parallel for
  for(int i = 0; i < Dim; i++)
    pp_vec[ i ] *= damp_op[ i ] * damp_op[ i ];
  */

  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim),
    eig_pp_vec(pp_vec, Dim);
  
  Eigen::Map<Eigen::Vector<double,-1>>   dmpt_op(damp_op,Dim);
  

  if(H_.size()>0)
    eig_vec = 2.0 * H_ * eig_p_vec - eig_pp_vec;
  
  if(Hc_.size()>0)
    eig_vec = 2.0 * Hc_ * eig_p_vec - eig_pp_vec;
  
    
  eig_pp_vec = eig_p_vec;
  eig_p_vec = eig_vec;
}



void Read_Hamiltonian::damp ( r_type damp_op[]){

  set_damp_op(damp_op);
  
  int Dim = this->parameters().DIM_;
 
  SpMatrixXp Id(Dim,Dim), gamma(Dim,Dim);//, dis(Dim,Dim);  dis.setZero();
  Id.setIdentity();
  gamma = Id;



  

  
  #pragma omp parallel for
  for(int i=0; i<Dim;i++)
    gamma.coeffRef(i,i) *=damp_op[ i ];

  if(H_.size()>0)
    H_ = gamma*H_;
  if(Hc_.size()>0)
    Hc_ = gamma*Hc_;
  
}


void Read_Hamiltonian::setup_velOp(){
  std::ifstream inFile;
  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  inFile.open(run_dir+"operators/"+filename+".VX.CSR");

  //std::cout<<"  /Remember vx/vcx hack. /real part of the Velocity is being dumped on read;"<<std::endl<<std::endl;
    
    
  std::size_t DIM, NNZ;
  inFile>>DIM;
  inFile>>NNZ;
  

  indexType outerIndexPtr[DIM+1];
  indexType innerIndices[NNZ];
  type values[NNZ];


  for(std::size_t j=0; j<NNZ; j++){
    double re_part , im_part;
    inFile>>re_part, inFile>>im_part;
    values[j] = re_part + std::complex<r_type>(0,1) * type( im_part );
  }

  for(std::size_t j=0; j<NNZ; j++)  
    inFile>>innerIndices[j];

  for(std::size_t j=0; j<DIM+1; j++)  
    inFile>>outerIndexPtr[j];    

  
  inFile.close();
  vxc_= Eigen::Map<Eigen::SparseMatrix<type, Eigen::RowMajor, indexType > >(DIM,DIM,NNZ,outerIndexPtr, innerIndices,values);

};


void Read_Hamiltonian::update_dis ( r_type dis_vec[], r_type damp_op[]){
  int subDim = this->parameters().SUBDIM_;
  int C   = this->parameters().C_,
      W   = this->parameters().W_;

  set_dis(dis_vec);

  if(H_.size()>0){
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     H_.coeffRef(C*W + i, C*W +i) = damp_op[i] * b_/a_;
     
  
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     H_.coeffRef(C*W + i, C*W +i) += damp_op[i] * dis_vec[i]/a_;
  }


  if(Hc_.size()>0){
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     Hc_.coeffRef(C*W + i, C*W +i) = damp_op[i] * b_/a_;
     
  
  #pragma omp parallel for
  for(int i=0; i<subDim;i++)
     Hc_.coeffRef(C*W + i, C*W +i) += damp_op[i] * dis_vec[i]/a_;
  }
  
}



/*
void Read_Hamiltonian::setup_velOp(){
  
  int Dim = this->parameters().DIM_,
      C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;

  vx_.resize(Dim,Dim);
  vx_.setZero();

  typedef Eigen::Triplet<r_type> T;

  std::vector<T> tripletList;
  tripletList.reserve(5*Dim);

  MatrixXp coordinates = coordinates_.data();


  for (int k=0; k<H_.outerSize(); ++k)
    for (typename SpMatrixXp::InnerIterator it(H_,k); it; ++it)
    {

      int i=it.row(),
	j=it.col();

      bool isJ_dev = false,
	   isI_dev = false;
      

      if(j>=C*W  &&  j<(C+Le)*W)
          isJ_dev = true;
      

      if(i>=C*W  &&  i<(C+Le)*W)
          isI_dev = true;

     
      if(isJ_dev && isI_dev){
	r_type ijHam = it.value(), v_ij;

 
	v_ij  =  ( coordinates(0,i) - coordinates(0,j) ) * ijHam;
        tripletList.push_back(T(i,j, v_ij) );
    }
  }


  vx_.setFromTriplets(tripletList.begin(), tripletList.end(),[] (const r_type &,const r_type &b) { return b; });  


    

  if(print_CSR){
    auto start_wr = std::chrono::steady_clock::now();    

    Eigen::SparseMatrix<type,Eigen::ColMajor> printVX(vx_.cast<type>());
    vx_.makeCompressed();
  
    int nnz = printVX.nonZeros(), cols = printVX.cols();
    type * valuePtr = printVX.valuePtr();//(nnz)
    int * innerIndexPtr = printVX.innerIndexPtr(),//(nnz)
      * outerIndexPtr = printVX.outerIndexPtr();//(cols+1)
  
    std::ofstream data2;
    data2.open("ARM.VX.CSR");

    data2.setf(std::ios::fixed,std::ios::floatfield);
    data2.precision(3);

    data2<<cols<<" "<<nnz<<std::endl;

    for (int i=0;i<nnz;i++)
      data2<<real(valuePtr[i])<<" "<<imag(valuePtr[i])<<" ";

    data2<<std::endl;
    for (int i=0;i<nnz;i++)
      data2<<innerIndexPtr[i]<<" ";

  
    data2<<std::endl;
    for (int i=0;i<cols;i++)
      data2<<outerIndexPtr[i]<<" ";

  
    data2.close();
    auto end_wr = std::chrono::steady_clock::now();


  std::cout<<"   Time to write vel. OP on disk:     ";
  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>
    (end_wr - start_wr).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "
           <<std::endl<<std::endl;

  }


}
*/
