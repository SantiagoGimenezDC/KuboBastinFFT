#include<iostream>
#include<fstream>
#include<chrono>
#include "Read_ConTable.hpp"
#include<fstream>
#include"../Kubo_solver/time_station.hpp"


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





};



void Read_ConTable::build_Hamiltonian(){


    std::size_t max_NN = connTable_.cols();
    std::size_t DIM = parameters().DIM_;

    typedef Eigen::Triplet<r_type> T;
    std::vector<T> tripletList;
    tripletList.reserve(35*DIM);

    
    Coordinates coords = coordinates();
    //Eigen::MatrixXd X(DIM, max_NN), Y(DIM, max_NN),Z(DIM, max_NN);

    //Eigen::Vector2d K(0,0);
    
    
    double V0pi_ = -2.7;//eV
    double V0sigma_ = 0.3675;//eV
    double qpibya0_ = 2.218;//A-1
    double qsigmabyb0_ = qpibya0_;
    double a0_ = 1.42;//A
    double d0_ = 3.43;//A
    double r0_ = 6.14;//A
    double lambdac_ = 0.265;//A



    for (std::size_t i = 0; i < DIM; i++) {


      for (std::size_t j = 0; j < max_NN; j++) {
	

	std::size_t j_ele = connTable_(i,j);
	if( j_ele >= DIM ) break;
	
	Eigen::Vector3d dist = Eigen::Vector3d::Zero(),
	  pos_i = coords.data().row(i),
	  pos_j = coords.data().row(j_ele);	  
	  
	dist(0) = ( pos_j(0) - pos_i(0) );
	dist(1) = ( pos_j(1) - pos_i(1) );
	dist(2) = ( pos_j(2) - pos_i(2) );

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
	  
	  tripletList.push_back(T(i,j_ele , hopping ) );
      	  
	  
	  //vals_[ i * max_NN + j ] = cosphi * cosphi * Vsigma  +  sinphi * sinphi * Vpi;
	  //rows_[ i * max_NN + j ] = i; 
	  //cols_[ i * max_NN + j ] = connTable_(i,j);
	  //vals_[ i * max_NN + j ]
	  
	}
    }




    H().resize(DIM,DIM);	
    H().setFromTriplets(tripletList.begin(), tripletList.end());

    //H().prune(1E-12);
    H().makeCompressed();

};


