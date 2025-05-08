#ifndef TBG_H
#define TBG_H

#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/SparseCore>

#include "../../static_vars.hpp"
#include "../Coordinates.hpp"
#include "../Device.hpp"
#include<vector>


class TBG: public Device{
  
  typedef Eigen::Matrix<type, -1, 1>                 VectorXdT;
  typedef Eigen::Matrix<type, -1, -1>               MatrixXdT;
  typedef Eigen::SparseMatrix<type,Eigen::RowMajor> SpMatrixXdT;
  
  typedef Eigen::Triplet<type> TdT;

  

  
  typedef Eigen::Matrix<std::complex<r_type>, -1, 1>  VectorXpc;
  typedef Eigen::Matrix<r_type, -1, 1>                VectorXp;
  typedef Eigen::Matrix<r_type, -1, -1>               MatrixXp;
  typedef Eigen::SparseMatrix<r_type,Eigen::RowMajor> SpMatrixXp;
  
  typedef Eigen::Triplet<r_type> T;

  
  private:
    int singleLayerDim_, fullLe_;
  
    Coordinates all_coordinates_,
    top_coordinates_,
    bottom_coordinates_;
  

    SpMatrixXp H_, vx_;
    r_type Emin_, Emax_, a_, b_;
  
    const r_type  a0_      = 0.142;
    const r_type  d0_     = 0.335;
    const r_type  VppPI_  = -2.7;
    const r_type  VppSIG_ = 0.48;
    const r_type  delta_  = 0.184*a0_*sqrt(3.0);
    
  public:
    virtual ~TBG(){};
    TBG(device_vars&);

    
    SpMatrixXp& Hamiltonian(){return H_;};
  
    virtual void setCoordinates();
    void bottom_layer_coordinates();
    void top_layer_coordinates();


    virtual r_type Hamiltonian_size(){ return ( H_.innerSize() + H_.nonZeros() + H_.outerSize() ) * sizeof(r_type); };

    virtual void build_Hamiltonian() {SlaterCoster_Hamiltonian(H_);  };
    virtual void adimensionalize(r_type, r_type);
    virtual void damp ( r_type*);
    virtual void update_dis(r_type*, r_type*); 

    virtual void rearrange_initial_vec(type*); //very hacky
    virtual void traceover(type*, type*, int, int) ;

  
    virtual void update_cheb_filtered ( type*, type*, type*, r_type*, r_type* , type);  
  
    virtual void H_ket ( type*, type*);
    virtual void H_ket ( type* vec, type* p_vec, r_type*, r_type*){this->H_ket(vec, p_vec);};  
    virtual void update_cheb ( type*, type*, type*, r_type*, r_type* );
    virtual void update_cheb ( type vec[], type p_vec[], type pp_vec[] ){ update_cheb(vec, p_vec, pp_vec, damp_op(), NULL); };  
    virtual void vel_op (type*, type*);
    virtual void vel_op (type* vec, type* p_vec, int){vel_op (vec, p_vec);};
    virtual void setup_velOp();
  
    void naiveHamiltonian (SpMatrixXp&);


  
    void SlaterCoster_Hamiltonian   (SpMatrixXp &);
    void intralayerNeighbours_SCH   (std::vector<T> &);
    void interlayerNeighbours_SCH   (std::vector<T> &);

  
    inline r_type SlaterCoster_intralayer_coefficient( r_type d_ij){
      return VppPI_  * exp( - ( d_ij - a0_) / delta_ );
    };

    inline r_type SlaterCoster_coefficient(Eigen::Matrix<r_type,3,1> R_ij, r_type d_ij){
      return VppPI_  * exp(-(d_ij-a0_)/delta_) * (1.0-pow(R_ij.dot(Eigen::Matrix<r_type,3,1>(0.0,0.0,1.0))/d_ij,2.0))+
	     VppSIG_ * exp(-(d_ij-d0_)/delta_) *      pow(R_ij.dot(Eigen::Matrix<r_type,3,1>(0.0,0.0,1.0))/d_ij,2.0);
    };



  
};




#endif // TBG_H
