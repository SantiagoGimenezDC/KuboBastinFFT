#ifndef STATES_HPP
#define STATES_HPP

#include <vector>
#include "Device/Device.hpp"
#include<iostream>
//#include "../solver_vars.hpp"

template<typename T>
class State{
private:
  int D_;
  T* data_;

public:
  ~State(){delete []data_;};

  State() : D_(0){};
  
  State(int D) : D_(D){
    data_ = new T[D_];
  };

  State(int D, T* data) : D_(D){
    data_ = new T[D_];

#pragma omp parallel for
    for(int i = 0; i < D; i++ )
      data_[i] = data[i];
    
  };
  
  // Copy constructor
  State(const State& other) : D_(other.D_), data_(new T[other.D_]) {
    std::copy(other.data_, other.data_ + D_, data_);
  }
  
  
  int D(){ return D_; };

  T* data(){ return data_; };

  T operator() (int i){ return data_[ i ]; };

  inline
  T operator[] (int i){ return data_[ i ]; };


  T operator*(State<T> vec){
    T result = 0.0;

    
    #pragma omp parallel
    {
        T local_result = 0.0;

        #pragma omp for
        for (int i = 0; i < D_; i++) {
            local_result += std::conj(data_[i]) * vec[i];
        }

        #pragma omp critical
        {
            result += local_result;
        }
    }
        
    return result;
  };
    
  void operator = ( State<T>& other_state){
        if ( this == &other_state ) 
            return ; // Handle self-assignment
        

        // Delete existing data if necessary	
        //if(D_>0)
	  //delete[] data_;

        D_ = other_state.D();
        data_ = new T[D_];

#pragma omp parallel for
        for (int i = 0; i < D_; i++) 
            data_[i] = other_state[i];
        
  };
  /*
 State operator+(const State& other) const {
        if (D_ != other.D_) {
            throw std::invalid_argument("Dimensions do not match.");
        }
        State result(D_);
#pragma omp parallel for
        for (int i = 0; i < D_; i++) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
	}*/
  
};




template <class State_T>
class States_buffer{
  typedef int indexType;
private:
  indexType D_, M_;
  std::vector<State_T*> states_buffer_;
    
public:
  States_buffer( indexType D, indexType M ) : D_(D), M_(M) {
    states_buffer_.resize(M);

    for( int m = 0; m < M_; m++ )
      states_buffer_.at(m) = new State_T(D_);
  };
  
  void push_back( State_T& new_state){ states_buffer_.push_back(new_state); };

  indexType memory_size()       { return M_ * states_buffer_.at(0).memory_size(); };
  std::vector<State_T*>* data() { return states_buffer_.data(); };
  
  State_T& operator()(indexType m ){ return *states_buffer_.at(m); };

  inline
  State_T& operator[](indexType m ){ return *states_buffer_[m]; };
  
};




class State_raw{
  typedef int indexType;
private:
  indexType D_ = 0;
  type* data_ = NULL;

public:
  ~State_raw(){};

  State_raw() : D_(0){};
  
  //  State_raw(int D) : D_(D){//This can be used for generating random vector
    //    data_ = new type[D_];
  //};

  State_raw(int D, type* data) : D_(D){//This works as a wrapper. Data must have been allocated previously.
    data_ = data;    
  };

  
  // Copy constructor
  State_raw(State_raw& other) : D_( other.D() ), data_(new type[ other.D() ]) {

    type* other_data = other.data();
    
#pragma omp parallel for
    for( int i=0; i<D_; i++ )
      data_[i] = other_data[i];

  };

  inline
  type cdot( State_raw& other ){
    type* other_data = other.data();

    type sum=0.;
    int i;
    
  # pragma omp parallel shared(sum)
    {
      std::complex< double > priv_sum = 0.;
      # pragma omp for
      for (i=0; i<D_;i++ )
      {
        priv_sum += std::conj<double>(data_[i]) * other_data[i];
      }

      #pragma omp critical
      {
        sum += priv_sum;
      }
    }
  return sum;
  };
  
  
  indexType D(){ return D_; };
  type* data(){ return data_; };

  type operator() (int i){ return data_[ i ]; };
  
  inline
  type operator[] (int i){ return data_[ i ]; };
    
  void operator = ( State_raw& other_state){//This cosiders that the data has been allocated in both States.
        if ( this == &other_state ) 
            return ; // Handle self-assignment
        if( other_state.D() != D_){
	  std::cout<<"You have asigned a state with an incorrect size"<<std::endl;
	    return;
	}
#pragma omp parallel for
        for (int i = 0; i < D_; i++) 
            data_[i] = other_state[i];
        
  };
};


  
class States_buffer_raw{
  typedef int indexType;
private:
  indexType D_=0, M_=0;
  type* raw_buffer_ = NULL;

  State_raw **states_ = NULL;

  
public:
  States_buffer_raw(){};
  ~States_buffer_raw(){
    if(D_>0){
      delete []states_;
      delete raw_buffer_;
    }
  };
  
  States_buffer_raw( indexType D, indexType M ) : D_(D), M_(M) {
    raw_buffer_ = new type[ D_ * M_ ];
    states_ = new State_raw*[ M_ ];

    for( int i = 0; i < M_; i++ )
      states_[i] = new State_raw( D, &(raw_buffer_[i]) );

  };

  States_buffer_raw( States_buffer_raw& copied ) {

    if( M_ != 0 )
      delete []raw_buffer_;

    D_ = copied.D();
    M_ = copied.M();

    
    raw_buffer_ = new type[ D_ * M_ ];

    type* copied_data = copied.data();

#pragma omp parallel for
    for( int i=0; i < M_ * D_; i++ )
      raw_buffer_[i] = copied_data[i];

    
  };

  indexType D(){ return D_; };
  indexType M(){ return M_; };
  
  indexType memory_size()       { return M_ * D_ * sizeof(type); };

  type* data() { return raw_buffer_; };

  inline
  State_raw& operator()(indexType m ){ return   *(states_[ m ]); };

  
  inline
  type* operator[](indexType m ){ return & ( raw_buffer_[ m * D_ ]); };

  inline
  type& operator()(indexType m, indexType i ){ return raw_buffer_[ m * D_ + i ] ; };



  States_buffer_raw& operator=( States_buffer_raw& copied ) {

    if( this == &copied )
      return *this;
    
    if( M_ != 0 )
      delete []raw_buffer_;

    D_ = copied.D();
    M_ = copied.M();

    
    raw_buffer_ = new type[ D_ * M_ ];

    type* copied_data = copied.data();

#pragma omp parallel for
    for( int i=0; i < M_ * D_; i++ )
      raw_buffer_[i] = copied_data[i];

    return *this; 
  };


};





template<class State_T>
class States_buffer_sliced: public States_buffer<State_T>{//MAKE sure that num_buffers = mem/col_size. Otherwise, you are exposed to M / num_buffers< M%num_buffers
private:
  int DIM_,  num_buffers_, buffer_size_, rest_size_;
  
public:
  States_buffer_sliced(int DIM, int M, int num_buffers) :
    DIM_(DIM), num_buffers_(num_buffers), buffer_size_( DIM / num_buffers ), rest_size_( DIM % num_buffers ), States_buffer<State_T>( DIM/num_buffers, M ){};

  
  int buffer_end(int s){
    if( s == num_buffers_ )
      return rest_size_;
    else
      return buffer_size_;
  };

};




class Chebyshev_states{
private:
  Device& device_;
  int head_num_ = 0;

  type *ket, *p_ket, *pp_ket;
  int D_;

  
public:
  //  Chebyshev_states(){};
  
  Chebyshev_states(Device& device ):device_(device), D_(device.parameters().DIM_){
    ket = new type[D_];
    p_ket = new type[D_];
    pp_ket = new type[D_];
  };
  
  ~Chebyshev_states(){
    delete []ket;
    delete []p_ket;
    delete []pp_ket;
  };


  type* operator()(int m ){
    if ( m == 0 )
      return pp_ket;
    else if ( m == 1 )
      return p_ket;
    else if ( m == 2 )
      return ket;
    else
      return NULL;
  };

  
  int update() {
    
    if( head_num_ == 0 )
      device_.H_ket( p_ket, pp_ket );

    else
      device_.update_cheb( ket, p_ket, pp_ket );

      
    head_num_++;
    return head_num_;
  };

  type* head(){ return ket; };
  
  void reset( type* init_state ){

    for(int i=0;i<D_;i++)
      pp_ket[i] = init_state[i];
    
    head_num_ = 0;
  };

};



/*
template<class State_T>
class Chebyshev_states: public States_buffer<State_T>{
private:
  Device& device_;
  int head_num_ = 0;

public:
  Chebyshev_states(Device& device ):device_(device), States_buffer<State_T>( device.parameters().DIM_, 3 ) {};

  
  int update() {
    
    if( head_num_ == 0 )
      device_.H_ket( (*this)(1).data(), (*this)(0).data() );

    else
      device_.update_cheb( (*this)(2).data(), (*this)(1).data(), (*this)(0).data() );

      
    head_num_++;
    return head_num_;
  };

  State_T& head(){ return (*this)(2); };
  
  void reset( State_T& init_state ){
    (*this)(0) = init_state;
    head_num_ = 0;
  };

  };*/






#endif //STATES_HPP
