#ifndef STATES_HPP
#define STATES_HPP

#include <vector>
#include "../../Device/Device.hpp"
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

  State(int D, T data) : D_(D){
    data_ = new T[D_];

#pragma omp parallel for
    for(int i = 0; i < D; i++ )
      data_[i] = data(i);
    
  };

  
  State(State& other_state):D_(other_state.D()){
    data_ = new T[D_];

#pragma omp parallel for
    for(int i = 0; i < D; i++ )
      data_[i] = other_state(i);
    
  };

  
  
  int D(){ return D; };
  T* data(){ return data_; };
  T& operator() (int i){ data_[ i ]; };

  
  State<T>& operator=(const State<T>& other_state){
        if (this == &other_state) 
            return *this; // Handle self-assignment
        

        // Delete existing data if necessary
        delete[] data_;

        D_ = other_state.D();
        data_ = new T[D_];

#pragma omp parallel for
        for (int i = 0; i < D_; i++) 
            data_[i] = other_state(i);
        

        return *this;
  };
  
};




template <class State_T>
class States_set{
  typedef int indexType;
private:
  indexType D_, M_;
  std::vector<State_T*> states_set_;
    
public:
  States_set( indexType D, indexType M ) : D_(D), M_(M) {
    states_set_.reserve(M);

    for( int m = 0; m < M_; m++ )
      states_set_(m) = new State_T(D_);
  };
  
  void push_back( State_T& new_state){ states_set_.push_back(new_state); };

  indexType memory_size()       { return M_ * states_set_.at(0).memory_size(); };
  std::vector<State_T*>& data() { return states_set_.data(); };
  
  State_T& operator()(indexType m ){ return states_set_.at(m); };
  State_T& operator[](indexType m ){ return states_set_[m]; };
  
  

};




template<class State_T>
class Chebyshev_states: public States_set<State_T>{
private:
  Device& device_;
  int head_num_ = 0;

public:
  Chebyshev_states(Device& device ):device_(device), States_set<State_T>( device_.parameters().DIM_, 3 ) {};

  int update() {
    
    if( head_num_ == 0 )
      (*this)(1) = device_.H_ket( (*this)(0).state_data().data() );

    else{
      (*this)(2) = 2 * device_.H_ket( (*this)(1).state_data().data() ) - (*this)(0);

      (*this)(0) = (*this)(1);
      (*this)(1) = (*this)(2);
    }
    
    head_num_++;
    return head_num_;
  };


  void reset( State_T& init_state ){
    (*this)(0) = init_state;
    head_num_ = 0;
  };


};


#endif //STATES_HPP
