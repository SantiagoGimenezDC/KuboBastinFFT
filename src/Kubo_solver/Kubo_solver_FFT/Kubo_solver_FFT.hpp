#ifndef KUBO_BASTIN_SOLVER_HPP
#define KUBO_BASTIN_SOLVER_HPP

#include<string>
#include<iostream>
#include<eigen-3.4.0/Eigen/Core>

#include"../../static_vars.hpp"
#include "../../Device/Device.hpp"
#include "../../Device/Graphene.hpp"
#include "../../kernel.hpp"
#include "../../vec_base.hpp"

#include "../solver_vars.hpp"

#include "../../States/States.hpp"





/*
class FFT_Operations{
private:
public:
  void preFactors();
  void energy_dependent_ops();
  void compute (States_buffer, States_buffer, std::vector<r_type>, int);

};
*/


template<class StateType>
class Chebyshev_states: public States_buffer<StateType>{
private:
  Device device_;
  int head_num_ = 0;

public:
  Chebyshev_states(Device& device ):device_(device), States_buffer<StateType>( device_.parameters().DIM_, 3 ) {};

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


  void reset( StateType& init_state ){
    (*this)(0) = init_state;
    head_num_ = 0;
  };


};



class Kubo_solver_FFT{
private:
  solver_vars parameters_;
  formula sym_formula_;

  
  Device&  device_;
  Kernel*   kernel_;
  CAP*      cap_;
  Vec_Base* vec_base_;

  

  
//---------------Large vectors----------------//  
  //Auxiliary - disorder and CAP vectors
  type *rand_vec_;
  r_type *dmp_op_,
         *dis_vec_ ;
//--------------------------------------------//


  
  
//---------------Dataset vectors--------------//
  std::vector<r_type> E_points_, conv_R_;
  std::vector<type> r_data_, final_data_ ;
//--------------------------------------------//

  
public:
  Kubo_solver_FFT();
  Kubo_solver_FFT( solver_vars&, Device&);
  ~Kubo_solver_FFT();
  
  solver_vars& parameters(){ return parameters_; };

  //Initializers
  void allocate_memory();
  void reset_recursion_vectors();
  void reset_Chebyshev_buffers();
  void reset_r_data(){ std::fill(r_data_.begin(), r_data_.end(), 0); };

  //Heavy duty
  void compute();

  
  void polynomial_cycle(Eigen_states_buffer<type>, Chebyshev_states<Eigen_state<type> > , int, bool );

  
  void Greenwood_FFTs__imVec ( Eigen_states_buffer<type>&, Eigen_states_buffer<type>&,  std::vector<type>&, int);
  void Bastin_FFTs ( type**, type**, type*, int);



  
  //Post-process
  void Greenwood_postProcess (  int  );
  void Bastin_postProcess ( int  );

  void integration ( r_type*, r_type*, r_type* );
  void partial_integration ( r_type*, r_type*, r_type* );

  void rearrange_crescent_order( r_type* );
  void eta_CAP_correct(r_type*, r_type* );  
  void plot_data   ( std::string, std::string );
  
};






class Kubo_solver_FFT_postProcess{//will interpret data_set of points k=0,...,nump-1 as associated to  the energies e_k=a_*cos(MPI*(2*k+0.5))-b_; 

        public:
        typedef double  r_value_t;
        typedef std::complex< r_value_t > value_t;

        Kubo_solver_FFT_postProcess(Kubo_solver_FFT&);
        ~Kubo_solver_FFT_postProcess(){};

  
        void operator()(const value_t*, const value_t*, int);
  
        void Greenwood_postProcess (const value_t*, const value_t*, int  );
        void Bastin_postProcess (const value_t*, const value_t*, int  );

        void integration ( const r_value_t*, const r_value_t*, r_value_t* );
        void integration_linqt ( const r_value_t*, const r_value_t*, r_value_t* );
        void partial_integration ( const r_value_t*, const r_value_t*, r_value_t* );

        void rearrange_crescent_order( r_value_t* );
        void eta_CAP_correct(r_value_t*, r_value_t* );  
        void plot_data   ( const std::string&, const std::string& );

        private:
        Kubo_solver_FFT parent_solver_;
         std::vector<r_type> E_points_;
        //           *conv_R_;

    
  };


#endif //KUBO_BASTIN_SOLVER_HPP


/*
  
  void Bastin_FFTs__reVec_noEta_2     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta_2 ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  
  void Bastin_FFTs__reVec_noEta     ( r_type**, r_type**, r_type*, r_type*);
  void Bastin_FFTs__imVec_noEta     ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);
  void Bastin_FFTs__imVec_eta       ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);

  void Greenwood_FFTs__reVec_noEta ( r_type**, r_type**, r_type*, r_type*);  
  void Greenwood_FFTs__reVec_eta   ( r_type*, r_type*, r_type*, r_type*);
  
  void Greenwood_FFTs__imVec_eta ( std::complex<r_type>**, std::complex<r_type>**, r_type*, r_type*);




*/
