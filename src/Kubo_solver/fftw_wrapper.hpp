#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>
#include<fftw3.h>


#include "../static_vars.hpp"

enum fft_direction{
  BACKWARD = FFTW_BACKWARD,
  FORWARD  = FFTW_FORWARD
};


class in_place_dft{
private:
  fftw_plan plan_;      
  fftw_complex * output_;
  
  int nump_;
  fft_direction dir_;

public:
  in_place_dft( int nump, fft_direction dir ) : nump_(nump), dir_(dir){
    
    output_ = ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * nump_ );    

  };

  
  inline void create() {
    void fftw_plan_with_nthreads(int nthreads = 1);

    plan_ = fftw_plan_dft_1d(nump_, output_, output_, dir_, FFTW_ESTIMATE); };//NOT thread safe

  inline void execute(){ fftw_execute( plan_ ); };

  
  inline std::complex<r_type>* output() { return reinterpret_cast< std::complex<r_type>* > ( output_ ); };  

  ~in_place_dft(){
    fftw_free(output_);
    fftw_destroy_plan(plan_);
  }
};




class out_of_place_dft{
private:
  fftw_plan plan_;      
  fftw_complex * input_,
               * output_;
  
  int nump_;
  fft_direction dir_;

public:
  out_of_place_dft( int nump, fft_direction dir ) : nump_(nump), dir_(dir){
    
    output_ = ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * nump_ );    
    input_  = ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * nump_ );    
    
    for(int m=0;m<nump;m++){
      input_ [ m ][ 0 ] = 0;
      input_ [ m ][ 1 ] = 0;  
    }

  };

  
  inline void create()  {
    void fftw_plan_with_nthreads(int nthreads = 1);
    plan_ = fftw_plan_dft_1d(nump_, input_, output_, dir_, FFTW_ESTIMATE);
  };//NOT thread safe
  inline void execute() { fftw_execute( plan_ ); };


  
  inline std::complex<r_type>* input()  { return reinterpret_cast< std::complex<r_type>* > ( input_ ); };    
  inline const std::complex<r_type>* output() { return reinterpret_cast< std::complex<r_type>* > ( output_ ); };


  
  inline const std::complex<r_type>* operator()() { return reinterpret_cast< std::complex<r_type>* > ( output_ ); };
  inline const std::complex<r_type>& operator()(int m) { return * reinterpret_cast< std::complex<r_type>* > ( &output_[m] ); };

  inline std::complex<r_type>* input(int m)  { return reinterpret_cast< std::complex<r_type>* > ( &input_[m] ); };    

  ~out_of_place_dft(){
    fftw_free(output_);
    fftw_free(input_);
    fftw_destroy_plan(plan_);
  }
};






enum cft_type{
  REDFT01 = FFTW_REDFT01
};

class out_of_place_cft{
private:
  fftw_plan
    plan_;

  r_type
    * input_,
    * output_;
   
  
  int nump_;
  cft_type type_;

public:
  
  out_of_place_cft( int nump, cft_type type ) : nump_(nump), type_(type){


    output_ = ( r_type* ) fftw_malloc( sizeof( r_type ) * nump_ );    
    input_  = ( r_type* ) fftw_malloc( sizeof( r_type ) * nump_ );    

    for(int m=0;m<nump;m++)
      input_ [ m ] = 0;
      
    
  };

  
  inline void create(){
    void fftw_plan_with_nthreads(int nthreads = 1);
    plan_ = fftw_plan_r2r_1d(nump_, input_, output_, FFTW_REDFT01, FFTW_ESTIMATE);
  };//NOT thread safe

  
  inline void execute() { fftw_execute( plan_ ); };

  

  
  inline r_type* input()  { return input_; };    
  inline const r_type* output() { return output_; };


  
  inline const r_type* operator()() { return output_; };
  inline const r_type& operator()(int m) { return output_[m] ; };

  inline std::complex<r_type>* input(int m)  { return reinterpret_cast< std::complex<r_type>* > ( &input_[m] ); };    


  
  ~out_of_place_cft(){
    fftw_free(output_);
    fftw_free(input_);
    fftw_destroy_plan(plan_);

  }
  
};

