#include<random>
#include<complex>
#include"static_vars.hpp"
#include"vec_base.hpp"

#include<fstream>
#include<iostream>

void generate_vec(int C, int W, int LE, type rand_vec[], int seed, int r){

  int DIM = (2*C+LE)*W, SUBDIM = LE*W;
  
  std::mt19937 gen;
  gen.seed(seed+r);
  std::uniform_int_distribution<> dis(0, SUBDIM);
  int random_site=dis(gen);

  for(int m=0; m < DIM; m++)
    rand_vec[m] = 0;
  
  rand_vec[random_site+C*W] = 1.0;
  
}



void generate_vec_im(int C, int W, int LE, std::complex<r_type> rand_vec[], int seed, int r){

  int DIM = (2*C+LE)*W, SUBDIM = LE*W;
  
  std::mt19937 gen;
  gen.seed(seed+r);
  std::uniform_int_distribution<> dis(0, SUBDIM);

  for(int m=0; m < DIM; m++)
    rand_vec[m] = 0;



  for( int j=C*W; j<C*W+SUBDIM; j++ ){
    double    phase = 2.0 * M_PI * double(dis(gen)) / SUBDIM;
    rand_vec[j] = std::polar(1.0,phase)/sqrt(double(SUBDIM));
  }
}




void Complex_Phase_real::generate_vec_im( std::complex<r_type> rand_vec[], int ){

  int SUBDIM = this->parameters().SUBDIM_;


  for( int j = 0; j <  SUBDIM; j++ )
    rand_vec [j] = (2.0*this->rng().get() - 1.)*sqrt(3.0)/ sqrt(double(SUBDIM));   
  
}


void Complex_Phase_real::generate_vec_re( r_type rand_vec[], int ){

  int SUBDIM = this->parameters().SUBDIM_;


  for( int j = 0; j <  SUBDIM; j++ )
    rand_vec [j] = (2.0*this->rng().get() - 1.)*sqrt(3.0)/ sqrt(double(SUBDIM));   
  
}





void Complex_Phase::generate_vec_im( std::complex<r_type> rand_vec[], int ){

  int SUBDIM = this->parameters().SUBDIM_;


  for( int j = 0; j <  SUBDIM; j++ ){
    double phase = 2.0 * M_PI * this->rng().get() ;
    rand_vec [j] = std::polar( 1.0, phase ) / sqrt(double(SUBDIM));
  }
  
}

void Complex_Phase::generate_vec_re( r_type rand_vec[], int ){

  int SUBDIM = this->parameters().SUBDIM_;

  
  for( int j = 0; j <  SUBDIM; j++ )
    rand_vec [j] = (2.0*this->rng().get() - 1.)*sqrt(3.0)/ sqrt(double(SUBDIM));   
  
}








void Direct::generate_vec_im( std::complex<r_type> rand_vec[], int ){

  int DIM = this->parameters().DIM_,
    SUBDIM = this->parameters().SUBDIM_;


#pragma omp parallel for
  for(int m=0; m < DIM; m++)
    rand_vec[m] = 0;

  int random_site = SUBDIM * this->rng().get();
  
  rand_vec[random_site] = 1.0;
  
}



void Direct::generate_vec_re( r_type rand_vec[], int ){

  int DIM    = this->parameters().DIM_,
    SUBDIM = this->parameters().SUBDIM_    ;


#pragma omp parallel for
  for(int m=0; m < DIM; m++)
    rand_vec[m] = 0;

  int random_site = SUBDIM * this->rng().get();
  
  rand_vec[random_site] = 1.0;
  
}

