#ifndef STATE_HPP
#define STATE_HPP

#include<string>
#include<iostream>
#include<eigen-3.4.0/Eigen/Core>

#include "../static_vars.hpp"



//whats being passed as reference/ as value?
  //Type conversions.
  //operator overloading hmmm
template <typename T>
class State{
  typedef int indexType;
private:
  indexType D_;
  std::vector<T> state_;
  Eigen::Vector< T, Eigen::Dynamic> state_data_;

public:
  State(indexType D):D_(D){ state_.reserve(D); };
  State(State& new_state): D_( new_state.dimension() ), state_( new_state.data() ){};
  State(std::vector<T> new_data): D_( new_data.size() ), state_( new_data ){};

  indexType dimension(){return D_; };
  std::vector<T>& data() {return state_; }
  indexType memory_size(){return state_.size() * sizeof(type); };

  
  type operator () (indexType i ){ return state_.at(i); };

  type operator [] (indexType i ){ return state_[i]; };

  void operator = (State& new_state){ state_ = new_state.data(); };  


  State operator * ( const T a ){
    std::vector<T> result(state_);

    #pragma omp parallel for
    for(indexType i; i<D_;i++)
      result += a * state_[i]; 

      
    return State( result );
  };  

  
  State operator + (State& other_state){
    std::vector<T> result(state_);

    #pragma omp parallel for
    for(indexType i; i<D_;i++)
      result += other_state[i]; 

      
    return State( result );
  };  



  State operator - (State& other_state){
    std::vector<T> result(state_);

    #pragma omp parallel for
    for(indexType i; i<D_;i++)
      result -= other_state[i]; 

      
    return State( result );
  };
  
};


template <class StateType>
class States_buffer{
  typedef int indexType;
private:
  indexType D_, M_;
  std::vector<StateType> states_;
    
public:
  States_buffer(indexType D, indexType M):D_(D),M_(M) {
    states_.reserve(M);

    for( indexType m; m < M_; m++ )
      states_(m).reserve(D_);
    
  };
  
  void push_back( StateType& new_state){ states_.push_back(new_state); };

  indexType memory_size(){ return M_ * states_.at(0).memory_size(); };
  std::vector<StateType>& data(){return states_; };
  
  StateType& operator()(indexType m ){ return states_.at(m); };
  StateType& operator[](indexType m ){ return states_[m]; };
  
  

};




template <typename T>
class Eigen_state{
private:
  Eigen::Vector< T, Eigen::Dynamic> state_data_;

public:
  Eigen_state( int D){ state_data_.resize(D); };
  Eigen_state( Eigen_state& new_state) : state_data_( new_state.state_data() ){};
  Eigen_state( Eigen::Vector< T, Eigen::Dynamic>& new_state_data) : state_data_( new_state_data ){};  

  int dimension(){return state_data_.size(); };

  int memory_size(){return state_data_.size() * sizeof(T); };
  
  inline
  Eigen::Vector< T, Eigen::Dynamic>& state_data() { return state_data_; }

  

  
  inline
  type operator () (int i ){ return state_data_(i); };

  inline
  type operator [] (int i ){ return state_data_[i]; };

  inline
  void operator = (Eigen_state& new_state){ state_data_ = new_state.state_data(); };  

  inline //isn't this creating an extra copy? maybe axpy is the way to go really. What if I use Eigen_state& as output?? -- My understanding is that the result gets destroyed after the oveloaded function call ends.
  Eigen_state& operator * ( const T a ){ return Eigen_state( a * state_data_ ); };  

  inline
  Eigen_state& operator + (Eigen_state& other_state){ return State( this->state_data() + other_state.state_data() ); };  

  inline
  Eigen_state& operator - (Eigen_state& other_state){ return State( this->state_data() - other_state.state_data() ); };  

  
};



template <typename T>
class Eigen_states_buffer{
private:
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> state_list_;
  std::vector< Eigen_state<T> > state_wrapper_; //wraps each column of state_list_ as an Eigen_state
  
public:
  Eigen_states_buffer(int D, int M){
    state_list_.resize(D,M);
    state_wrapper_.reserve(M);
    
    for(int m = 0; m < state_list_.cols(); m++)
      state_wrapper_.push_back( Eigen_state( state_list_.col(m) ) );
    
  }

  inline
  int size(){return state_list_.cols(); };

  inline
  void reset(){state_list_.setZero();};

///implement size matching assertions?  
  inline
  void insert( Eigen_state<T>& new_state, int m){ state_list_.col(m) = new_state.state_data(); };

  double memory_size(){ return double(state_list_.cols()) * double( state_list_.rows() ) * sizeof(T); };
  
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data(){ return state_list_; };

  inline
  Eigen_state<T>& operator[](int m ){ return state_wrapper_[m]; };
  
};



#endif //STATE_H






/*

*/
