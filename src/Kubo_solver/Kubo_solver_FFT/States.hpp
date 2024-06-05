#ifndef STATES_HPP
#define STATES_HPP

#include <vector>
//#include "../solver_vars.hpp"

template<typename T>
class State{
private:
  int D_;
  T* data_;

public:
  ~State(){delete []data_;};

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
  
  void operator = ( State<T>& other_state){
        if ( this == &other_state ) 
            return ; // Handle self-assignment
        

        // Delete existing data if necessary
        delete[] data_;

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







#endif //STATES_HPP
