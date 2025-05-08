#include<iostream>
#include<fstream>
#include<complex>
#include<random>
#include<chrono>
#include<eigen3/Eigen/Core>


#include"Graphene.hpp"



void Graphene::damp_csc ( r_type damp_op[]){
  if(csc_mode){
  int Dim = this->parameters().DIM_;
 
  SpMatrixXp Id(Dim,Dim), gamma(Dim,Dim);//, dis(Dim,Dim);  dis.setZero();
  Id.setIdentity();
  gamma = Id;


  #pragma omp parallel for
  for(int i=0; i<Dim;i++)
    gamma.coeffRef(i,i) *=damp_op[ i ];

  H_ = gamma*H_;
  }
  else return;
}

void Graphene::update_cheb_csc ( type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[]){

  int Dim = this->parameters().DIM_,
      W = this->parameters().W_,
      C = this->parameters().C_,
      Le = this->parameters().LE_;
 
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


void Graphene::H_ket_csc ( type* vec, type* p_vec, r_type* dmp_op, r_type* dis_vec){
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




void Graphene::vel_op_csc (type vec[], type p_vec[]){
  int Dim = this->parameters().DIM_;
  
  Eigen::Map<VectorXdT> eig_vec(vec,Dim),
    eig_p_vec(p_vec, Dim);

  eig_vec = vx_ * eig_p_vec;
  
};




void Graphene::setup_velOp(){
  
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
  
  int nnz = printVX.nonZeros(), cols = printVX.cols(), rows = printVX.rows();
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



void Graphene::single_neighbor_Hamiltonian(){//ZIGZAG ON X DIR. AND ITS WRONG

  int W      = this->parameters().W_,
      Dim    = this->parameters().DIM_,
      fullLe = fullLe_;

  r_type e = 0.0,
         t = t_standard_;

  typedef Eigen::Triplet<r_type> T;

  std::vector<T> tripletList;
  tripletList.reserve(5*Dim);

  for(int n = 0; n < fullLe; n++){
    tripletList.push_back( T( n * W, n * W, e ));

    for(int k = 0; k < W; k++){

      if( k >= 1){//Diagonal Blocks
        tripletList.push_back( T( n * W + k, n * W + k, e ));
	    if( n % 2 == 0 ){
          tripletList.push_back( T( n * W + k, n * W + k-1, -t * ( k % 2 ) ) );
          tripletList.push_back( T( n * W + k-1, n * W + k, -t * ( k % 2 ) ) );
	    }
	    else{
          tripletList.push_back(T(n*W+k,n*W+k-1,-t*((k+1)%2)));
          tripletList.push_back(T(n*W+k-1,n*W+k,-t*((k+1)%2)));
	    }
    }

      if( n >= 1){//Off-diagonal blocks
          tripletList.push_back(T((n-1)*W+k,n*W+k,-t));
          tripletList.push_back(T(n*W+k,(n-1)*W+k,-t));
      }
    }
  }


  H_.resize(Dim, Dim);
  H_.setFromTriplets(tripletList.begin(), tripletList.end());


};



void Graphene::set_coordinates(){
  int W   = this->parameters().W_,
    fullLe = fullLe_;

  MatrixXp coordinates(3, fullLe*W);

  for(int j = 0; j < fullLe; j++){
    for(int i=0; i<W; i++){
      int n=j*W+i;

      coordinates(1,n)=-i*a0_*cos(M_PI/6.0);


      if(i%2==1){
        coordinates(0, n)=a0_*(
                               sin(M_PI/6.0) +
                               (j/2)*(1.0+2.0*sin(M_PI/6.0)) +
                               ((j+1)/2)
                               );
      }
      else{
        coordinates(0, n)=a0_*(
                               ((j+1)/2)*(1.0+2.0*sin(M_PI/6.0)) +
                               (j/2)
                               );

      }
    }
  }

  coordinates_.reset(coordinates);
}



void Graphene::SlaterCoster_Hamiltonian(){
  int W      = this->parameters().W_,
      Dim    = this->parameters().DIM_,
      Mdef   = 2 * this->parameters().d_min_,//2*d_min,
      Mdefi, j;

  r_type d_min  = this->parameters().d_min_*a0_;

  Eigen::Matrix<r_type,3,1> R_ij(0.0,0.0,0.0);

  MatrixXp coordinates = coordinates_.data();

  r_type d_ij,
         t_ij;


  typedef Eigen::Triplet<r_type> T;

  std::vector<T> tripletList;
  tripletList.reserve(5*Dim);



  auto start_RV = std::chrono::steady_clock::now();


  for(int i = 0; i < Dim; i++){
    for(int j_n = i/W; j_n <= i / W + Mdef && j_n < Dim / W; j_n++){

      if(j_n == i/W)
	      Mdefi = -1;
      else
	      Mdefi = Mdef;

      for(int i_n = (i%W-Mdefi>0 ? i%W-Mdefi : 0); i_n<=i%W+Mdef && i_n<W; i_n++){

	      j    = j_n*W+i_n;
        R_ij = coordinates.col(j)-coordinates.col(i);
        d_ij = R_ij.norm();

        if(d_ij < d_min){
	        t_ij = SlaterCoster_intralayer_coefficient(d_ij);
                 tripletList.push_back(T(i,j,t_ij));
                 tripletList.push_back(T(j,i,t_ij));
	      }
      }
    }
  }


  H_.resize(Dim, Dim);
  H_.setFromTriplets(tripletList.begin(), tripletList.end());


  auto end_RV = std::chrono::steady_clock::now();


  std::cout<<"   Time to run through the intralayer elements:     ";
  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>
    (end_RV - start_RV).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "
           <<std::endl<<std::endl;



  

  if(print_CSR){
  auto start_wr = std::chrono::steady_clock::now();    

  Eigen::SparseMatrix<type,Eigen::ColMajor> printH(H_.cast<type>());
  printH.makeCompressed();
  
  int nnz = printH.nonZeros(), cols = printH.cols(), rows = printH.rows();
  type * valuePtr = printH.valuePtr();//(nnz)
  int * innerIndexPtr = printH.innerIndexPtr(),//(nnz)
      * outerIndexPtr = printH.outerIndexPtr();//(cols+1)
  
  std::ofstream data2;
  data2.open("ARM.HAM.CSR");

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


  std::cout<<"   Time to write hamiltonian on disk:     ";
  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>
    (end_wr - start_wr).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "
           <<std::endl<<std::endl;

  }
  

}



