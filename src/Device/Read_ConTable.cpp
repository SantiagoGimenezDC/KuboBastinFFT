#include<iostream>
#include<fstream>
#include<chrono>
#include "Read_ConTable.hpp"
#include<fstream>
#include"../Kubo_solver/time_station.hpp"
#include<omp.h>

Read_ConTable::Read_ConTable(device_vars& device_vars):Read_Hamiltonian(device_vars){

  std::string run_dir  = parameters().run_dir_,
    filename = parameters().filename_;

  set_sysLength(1.0);  
  set_sysSubLength(1.0);
  
  std::ifstream inFile_xyz;  
  inFile_xyz.precision(14);
  std::cout<<"  Reading the connectivity table and xyz at: "<<run_dir+"operators/"+filename+".nntable and "<<  run_dir+"operators/"+filename+"_supercell.carbon"<<std::endl<<std::endl;
  

  inFile_xyz.open(run_dir+"operators/"+filename+"_supercell.carbon");

  
  std::size_t DIM;

  inFile_xyz>>U_(0,0);
  inFile_xyz>>U_(0,1);
  inFile_xyz>>U_(1,0);
  inFile_xyz>>U_(1,1);

  inFile_xyz>>a_cc_;  
  inFile_xyz>>num_cutoff_;
  inFile_xyz>>DIM;


  parameters().DIM_    = DIM;
  parameters().SUBDIM_ = DIM;  
  parameters().C_      = 0;
  parameters().W_      = 1;
  parameters().LE_     = DIM;




  
  MatrixXp coords(DIM, 3);
  
  for(std::size_t i=0; i<DIM; i++){
    for(int j=0; j<3; j++)  
      inFile_xyz>>coords(i,j);
  }  

  coordinates().reset(coords);


  
  std::ifstream inFile;
    inFile.open(run_dir+"operators/"+filename+".nntable");

  double throw_value;
  std::size_t  max_NN=1;
  while (inFile >> throw_value && inFile.peek() != '\n')  
    max_NN++;

  inFile.clear(); // Clear any error flags
  inFile.seekg(0, std::ios::beg); // Move the stream pointer to the beginning of the file

    
  connTable_.resize(DIM,max_NN);
  
  for(std::size_t i=0; i<DIM; i++){
    for(std::size_t j=0; j<max_NN; j++)  
      inFile>>connTable_(i,j);
  }  

  inFile.close();
  inFile_xyz.close();
  
  std::cout<<"  Finished Reading the connectivity table and xyz files;"<<std::endl<<std::endl;


  //  generate_Hamiltonian();
  //generate_velOp();


};



void Read_ConTable::build_Hamiltonian(){


    std::size_t max_NN = connTable_.cols();
    std::size_t DIM = parameters().DIM_;

    typedef Eigen::Triplet<r_type, indexType> T;
    std::vector<T> tripletList;
    tripletList.reserve( max_NN * DIM );

    
    MatrixXp coords = coordinates().data();
    //Eigen::MatrixXd X(DIM, max_NN), Y(DIM, max_NN),Z(DIM, max_NN);

    //Eigen::Vector2d K(0,0);
    
    std::size_t num_threads=omp_get_num_threads();

    //    double U0_norm = ( U_(0,0) * U_(1,1) - U_(1,0) * U_(0,1) ),
    // U1_norm = ( U_(0,0) * U_(1,1) - U_(1,0) * U_(0,1) );



 #pragma omp parallel 
  {
    int id,  Nthrds;
    std::size_t l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * max_NN / Nthrds;
    l_end   = (id+1) * max_NN / Nthrds;

    if (id == Nthrds-1)
      l_end = max_NN;


    std::vector<T> local_tripletList;
    local_tripletList.reserve( max_NN * DIM / num_threads );

    
    for (std::size_t j = l_start; j < l_end; j++) {
 
      for (std::size_t i = 0; i < DIM; i++) {

        std::size_t j_ele = connTable_(i,j);

        if( j_ele < DIM ){

	 
	Eigen::Vector3d dist = Eigen::Vector3d::Zero(),
	  pos_i = coords.row(i),
	  pos_j = coords.row(j_ele);	  
	
	dist = pos_j-pos_i;
	
	double atmp = ( dist(0) * U_(1,1) - dist(1) * U_(1,0) ) / ( U_(0,0) * U_(1,1) - U_(1,0) * U_(0,1) ),
	       btmp = ( dist(0) * U_(0,1) - dist(1) * U_(0,0) ) / ( U_(1,0) * U_(0,1) - U_(0,0) * U_(1,1) );          
	
	if( atmp > 0.5){
	  dist(0) -= U_(0,0);
	  dist(1) -= U_(0,1);
	}
	if( atmp < -0.5){
	  dist(0) += U_(0,0);
	  dist(1) += U_(0,1);
        }
	    
	if( btmp > 0.5){
	  dist(0) -= U_(1,0);
	  dist(1) -= U_(1,1);
	}
	if( btmp < -0.5){
	  dist(0) += U_(1,0);
	  dist(1) += U_(1,1);
        }
	  


	  dist *= a0_;
	  
	  double normi = sqrt ( dist(0) * dist(0) + dist(1) * dist(1) + dist(2) * dist(2) );
	  double Vpi = V0pi_ * exp ( qpibya0_ * a0_ * ( 1.0 - normi / a0_) ) / ( 1.0 + exp( ( normi - r0_ ) / lambdac_ ) );
	  double Vsigma = V0sigma_ * exp ( qsigmabyb0_ * d0_ * ( 1.0 - normi / d0_) ) / ( 1.0 + exp( ( normi - r0_ ) / lambdac_ ) );

	  
	  double cosphi = dist(2) / normi;
	  double sinphi = sqrt( 1.0 - cosphi * cosphi );

	  double hopping = cosphi * cosphi * Vsigma  +  sinphi * sinphi * Vpi;
	  
	  //hopping *= exp( std::complex<double>(0,1) * ( X(i,j) * K(0) + X(i,j) * K(1) ) );
	  
	  local_tripletList.push_back(T(i,j_ele , hopping ) );
      	  

	  
	}
    }
   }
   
    #pragma omp critical
    tripletList.insert(tripletList.end(), local_tripletList.begin(), local_tripletList.end());
  }



    H().resize(DIM,DIM);	
    H().setFromTriplets(tripletList.begin(), tripletList.end(), [] (const r_type &,const r_type &b) { return b; });
    //H().prune(1E-12);
    H().makeCompressed();
	 
};




void Read_ConTable::setup_velOp(){
  
  int Dim = this->parameters().DIM_;
  vx().resize(Dim,Dim);
  vx().setZero();

  typedef Eigen::Triplet<r_type,  indexType> T;

  std::vector<T> tripletList;
  tripletList.reserve( 5 * Dim);

  MatrixXp coords = coordinates().data();

  std::cout<<"  Generating VX from Hamiltonian"<<std::endl;
  for (int k=0; k<H().outerSize(); ++k)
    for (typename SpMatrixXp::InnerIterator it(H(),k); it; ++it)
    {

      int i = it.row(),
	  j = it.col();

      Eigen::Vector3d dist = Eigen::Vector3d::Zero(),
	  pos_i = coords.row(i),
	  pos_j = coords.row(j);	  
	  
      dist(0) = ( pos_j(0) - pos_i(0) );
      dist(1) = ( pos_j(1) - pos_i(1) );

	

      double atmp = ( dist(0) * U_(1,1) - dist(1) * U_(1,0) ) / ( U_(0,0) * U_(1,1) - U_(1,0) * U_(0,1) ),
	     btmp = ( dist(0) * U_(0,1) - dist(1) * U_(0,0) ) / ( U_(1,0) * U_(0,1) - U_(0,0) * U_(1,1) );          
	
	if( atmp > 0.5)
	  dist(0) -= U_(0,0);
	
	    
	if( atmp < -0.5)
	  dist(0) += U_(0,0);
	
	     
	if( btmp > 0.5)
	  dist(0) -= U_(1,0);
	
	    
	if( btmp < -0.5)
	  dist(0) += U_(1,0);
	
	  
	  

      r_type ijHam = it.value(), v_ij;

      r_type a0_ = 1.42;
      v_ij  =  a0_ * dist(0) * ijHam;
      tripletList.push_back(T(i,j, v_ij) );

  }


  vx().setFromTriplets(tripletList.begin(), tripletList.end(),[] (const r_type &,const r_type &b) { return b; });  
  vx().makeCompressed();
  std::cout<<"  Finished Generating VX from Hamiltonian"<<std::endl<<std::endl;
    
  bool print_CSR =false;
  if(print_CSR){
    auto start_wr = std::chrono::steady_clock::now();    

    Eigen::SparseMatrix<type,Eigen::ColMajor> printVX(vx().cast<type>());
    
  
    int nnz = printVX.nonZeros(), cols = printVX.cols();
    type * valuePtr = printVX.valuePtr();//(nnz)
    int * innerIndexPtr = printVX.innerIndexPtr(),//(nnz)
      * outerIndexPtr = printVX.outerIndexPtr();//(cols+1)
  
    std::ofstream data2;
    data2.open("ARM.VX.CSR");

    data2.setf(std::ios::fixed,std::ios::floatfield);
    data2.precision(18);

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
