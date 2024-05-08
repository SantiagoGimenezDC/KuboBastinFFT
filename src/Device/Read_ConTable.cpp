#include<iostream>
#include<fstream>
#include<chrono>
#include "Read_ConTable.hpp"
#include<fstream>



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


  generate_Hamiltonian();

  std::cout<<"  Finished building the Hamiltonian;"<<std::endl<<std::endl;
};


void Read_ConTable::generate_Hamiltonian(){


    std::size_t max_NN = connTable_.cols();
    std::size_t DIM = parameters().DIM_;

    Coordinates coords = coordinates();
    Eigen::MatrixXd X(DIM, max_NN), Y(DIM, max_NN),Z(DIM, max_NN);

    //Eigen::Vector2d K(0,0);
    
    
    double V0pi_ = -2.7;//eV
    double V0sigma_ = 0.3675;//eV
    double qpibya0_ = 2.218;//A-1
    double qsigmabyb0_ = qpibya0_;
    double a0_ = 1.42;//A
    double d0_ = 3.43;//A
    double r0_ = 6.14;//A
    double lambdac_ = 0.265;//A


    vals_.resize( DIM * max_NN );
    rows_.resize( DIM * max_NN );
    cols_.resize( DIM * max_NN );

    //    #pragma omp parallel for
    for (std::size_t i = 0; i < DIM; i++) {
      std::cout<<i<<"/"<<DIM<<std::endl;
      for (std::size_t j = 0; j < max_NN; j++) {
	  Eigen::Vector3d tmp_dist = Eigen::Vector3d::Zero();
	  
	  
	  tmp_dist(0) = ( coords.data()(connTable_(i,j), 0) - coords.data()(i,0) );
	  tmp_dist(1) = ( coords.data()(connTable_(i,j), 1) - coords.data()(i,1) );
	  Z(i,j) = ( coords.data()(connTable_(i,j), 2) - coords.data()(i,2) );

          double atmp = ( tmp_dist(0) * U_(1,1) - tmp_dist(1) * U_(1,0) ) / ( U_(0,0) * U_(1,1) - U_(1,0) * U_(0,1) ),
	         btmp = ( tmp_dist(0) * U_(0,1) - tmp_dist(1) * U_(0,0) ) / ( U_(1,0) * U_(0,1) - U_(0,0) * U_(1,1) );          

	  
	  if( atmp > 0.5){
	    X(i,j) = tmp_dist(0) - U_(0,0);
	    Y(i,j) = tmp_dist(1) - U_(0,1);
	  }
	  if( atmp < -0.5){
	    X(i,j) = tmp_dist(0) + U_(0,0);
	    Y(i,j) = tmp_dist(1) + U_(0,1);
	  }
	    
	  if( btmp > 0.5){
	    X(i,j) = tmp_dist(0) - U_(1,0);
	    Y(i,j) = tmp_dist(1) - U_(1,1);
	  }

	  if( btmp < -0.5){
	    X(i,j) = tmp_dist(0) + U_(1,0);
	    Y(i,j) = tmp_dist(1) + U_(1,1);
	  }


	  X(i,j) = a0_ * ( X(i,j) - coords.data()(i,0) );
	  Y(i,j) = a0_ * ( Y(i,j) - coords.data()(i,1) );
	  Z(i,j) = a0_ * ( Z(i,j) - coords.data()(i,2) );

	  
	  double normi = sqrt ( X(i,j) * X(i,j) + Y(i,j) * Y(i,j) + Z(i,j) * Z(i,j) );
	  double Vpi = V0pi_ * exp ( qpibya0_ * a0_ * ( 1.0 - normi / a0_) ) / ( 1.0 + exp( ( normi - r0_ ) / lambdac_ ) );
	  double Vsigma = V0sigma_ * exp ( qsigmabyb0_ * d0_ * ( 1.0 - normi / d0_) ) / ( 1.0 + exp( ( normi - r0_ ) / lambdac_ ) );

	  
	  double cosphi = Z(i,j) / normi;
	  double sinphi = sqrt( 1.0 - cosphi * cosphi );

	  
	  
	  vals_[ i * max_NN + j ] = cosphi * cosphi * Vsigma  +  sinphi * sinphi * Vpi;
	  rows_[ i * max_NN + j ] = i; 
	  cols_[ i * max_NN + j ] = connTable_(i,j);
	  //vals_[ i * max_NN + j ] *= exp( std::complex<double>(0,1) * ( X(i,j) * K(0) + X(i,j) * K(1) ) );
	  
	}
    }

    Eigen::Map<Eigen::SparseMatrix<r_type, Eigen::RowMajor> > sm1(DIM, DIM, DIM*max_NN, cols_.data(), rows_.data(), vals_.data() );

    set_H(sm1);
    H().prune(1E-9);
    H().makeCompressed();
};


