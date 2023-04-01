#include "Coordinates.hpp"
#include<iostream>
#include<fstream>


void Coordinates::plotSample(){
  std::ofstream data;
  data.open("./lattice_data.txt");
  data<<coordinates_.transpose(); 
  data.close();
}



void Coordinates::rotate( r_type theta){ //Rotation around the CURRENT OIGIN!!!
  int Dim = coordinates_.cols();
  
  Eigen::Matrix<r_type,2,2> Rotation;
  Rotation(0,0)=cos(theta);
  Rotation(0,1)=-sin(theta);
  Rotation(1,0)=sin(theta);
  Rotation(1,1)=cos(theta);

  for(int i=0; i<Dim;i++)
    coordinates_.block(0,i,2,1)=Rotation*coordinates_.block(0,i,2,1);

}

void Coordinates::centralize(){//Only works if sample begins at origin
  int Dim = W_*fullLe_;


  int lcM = W_/2;
  int lcN = fullLe_/2;

  origin_entries_(0) = lcN;
  origin_entries_(1) = lcM;


  Eigen::Matrix<r_type,1,2> delta = coordinates_.col( lcN * W_ + lcM ).segment(0,2);
  

  for(int i=0; i<Dim;i++)
    coordinates_.col(i).segment(0,2) -= delta;
  
}


void Coordinates::translate(Eigen::Matrix<r_type,3,1> &b){
  int W = W_,
      Dim = W_*fullLe_;
  
  int newCenter=0;
  Eigen::Matrix<r_type,3,1> closer(coordinates_.block(0,0,3,1)), dist=Eigen::Matrix<r_type,3,1>::Zero();
  
  for(int i=0;i<Dim;i++){
    dist = coordinates_.block(0,i,3,1)-(-b);
    if(dist.norm()<closer.norm()){
      newCenter = i;
      closer    = coordinates_.block(0,i,3,1);
    }
  }

  
  origin_entries_(0) = newCenter / W;
  origin_entries_(1) = newCenter % W;

  
  for(int i=0; i<Dim; i++)
    coordinates_.block(0,i,3,1) += b;
  //Atualizar originCoordinates_ de acordo com a translação!!!
}
