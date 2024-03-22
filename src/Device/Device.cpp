#include"Device.hpp"
#include<eigen-3.4.0/Eigen/Core>
#include<eigen-3.4.0/Eigen/SparseCore>
#include<chrono>
#include<iostream>



void Device::Anderson_disorder(r_type disorder_vec[]){

  int SUBDIM = device_vars_.SUBDIM_;
  r_type str = device_vars_.dis_str_;
  
  for(int i=0;i<SUBDIM; i++){
    r_type random_potential = str * this->rng().get()-str/2;

    disorder_vec[i] = random_potential;
  }
  
}


void Device::minMax_EigenValues( int maxIter, r_type& eEmax, r_type& eEmin){ //Power Method; valid if eigenvalues are real
  int DIM = device_vars_.DIM_;

  Eigen::Matrix<type, -1, 1> y = Eigen::Matrix<type, -1, 1>::Constant(DIM,1, 1.0/sqrt(DIM) ),
            y_Ant=y;

  r_type filler_vec_2[DIM], filler_vec[DIM];

  
  for(int k=0; k<DIM; k++){
    filler_vec[k] = 0;
    filler_vec_2[k] = 1.0;
  }

  r_type y_norm = 0;
  r_type Emax, Emin;

  std::cout<<"   Calculating Energy band bounds:    "<<std::endl;
  auto start = std::chrono::steady_clock::now();

  
  for( int i=0; i<maxIter; i++){
    
    this->H_ket(y.data(),y_Ant.data(), filler_vec_2, filler_vec);
    y_norm=y.norm();
    y=y/y_norm;
    y_Ant=y;
  }

  
  this->H_ket(y.data(),y_Ant.data());

  Emax = std::real(y_Ant.dot(y)/y_Ant.squaredNorm());



  
  y  =  Eigen::Matrix<type, -1, 1>::Constant(DIM,1, 1.0/sqrt(DIM) );

  this->adimensionalize(1.0,-Emax);
  
  for( int i=0; i<maxIter; i++){
    this->H_ket(y.data(),y_Ant.data(), filler_vec_2, filler_vec);
    y_norm=y.norm();

    y=y/y_norm;
    y_Ant=y;  
  }


  this->H_ket(y.data(),y_Ant.data());
  
  Emin  = std::real(((y_Ant.dot(y))/y_Ant.squaredNorm()));
  Emin += Emax;

  this->adimensionalize(1.0,Emax);
   
  auto end = std::chrono::steady_clock::now();
  std::cout<<"   Time to perform Lanczos Recursion:    ";
  int millisec0=std::chrono::duration_cast<std::chrono::milliseconds>
                (end - start).count();
  int sec0=millisec0/1000, min0=sec0/60, reSec0=sec0%60;
  std::cout<<min0<<" min, "<<reSec0<<" secs;"<<
	     " ("<< millisec0<<"ms) "<<std::endl<<std::endl;     


  
  eEmin  = std::min(Emax,Emin);
  eEmax  = std::max(Emax,Emin);
  
    
  std::cout<<"    Highest absolute energy bound:    "<<eEmax<<std::endl;
  std::cout<<"    Lower absolute energy bound:      "<<eEmin<<std::endl<<std::endl<<std::endl;
}
