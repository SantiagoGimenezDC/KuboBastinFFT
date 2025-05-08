#ifndef COORDINATES_H
#define COORDINATES_H

#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/SparseCore>

#include "../static_vars.hpp"

#include "Device.hpp"
#include<vector>

class Coordinates{
  typedef Eigen::Matrix<r_type, -1, 1>                VectorXp;
  typedef Eigen::Matrix<r_type, -1, -1>               MatrixXp;

private:
  int W_, LE_, C_, fullLe_;
  MatrixXp coordinates_;
  Eigen::Vector2i origin_entries_;
public:
  Coordinates(int W, int LE, int C):W_(W), LE_(LE),C_(C),fullLe_(LE+2*C){
   origin_entries_ = Eigen::Vector2i(0,0);
  };
  
  Coordinates():W_(0), LE_(0),C_(0),fullLe_(0){
   origin_entries_ = Eigen::Vector2i(0,0);
  };
  
  Coordinates(int W, int LE, int C, MatrixXp& coordinates):W_(W), LE_(LE),C_(C),fullLe_(LE+2*C), coordinates_(coordinates){
   origin_entries_ = Eigen::Vector2i(0,0);
  };

  void plotSample();
  void reset(MatrixXp& coordinates){ coordinates_ = coordinates; };
  MatrixXp data(){ return coordinates_; };
  Eigen::Vector2i origin_entries(){ return origin_entries_; };
  
  void resetOrigin()  { origin_entries_ = Eigen::Vector2i(0,0); }
  void centralize();
  void rotate( r_type );
  void translate( Eigen::Matrix<r_type,3,1>& );
};


#endif //COORDINATES_H
