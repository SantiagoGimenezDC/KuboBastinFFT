#ifndef TEST_HAMILTONIAN_HPP
#define TEST_HAMILTONIAN_HPP

#include<iostream>
#include "static_vars.hpp"
#include "update_cheb.hpp"

void print_hamiltonian(){
  type ket[DIM_],
       tmp[DIM_],
       tmp2[SUBDIM_],
    zero[DIM_],
    //    v[Dim*Dim],
    polys[SUBDIM_*M_];

  
  for(int k=0;k<DIM_;k++){
    ket[k] = 0.0;
    tmp[k] = 0.0;
    zero[k] = 0.0;
  }

  

  for(int j=0;j<DIM_;j++){
    ket[j] = 1.0;
    if(j>0)
      ket[j-1]=0.0;

    update_cheb(tmp,ket,zero,2*2.7,0);

    for(int i=0;i<DIM_;i++){
      if(tmp[i]!=0)
        std::cout<<tmp[i]<<"  ";
      else
	std::cout<<"   ";
    }
    std::cout<<std::endl;
  }

  std::cout<<std::endl;
  std::cout<<std::endl;


    
  for(int k=0;k<DIM_;k++){
    ket[k] = 0.0;
    tmp[k] = 0.0;
    zero[k] = 0.0;
  }

  

  for(int j=0;j<SUBDIM_;j++){
    ket[j] = 1.0;
    if(j>0)
      ket[j-1]=0.0;

    vel_op(tmp,ket);
    for(int i=0;i<SUBDIM_;i++){
      // v[i*subDim+j]=tmp2[i];
      if(tmp2[i]!=0)
        std::cout<<tmp2[i]/1.35<<"  ";
      else
	std::cout<<"   ";
    }
    std::cout<<std::endl;
  }





  for(int k=0;k<SUBDIM_;k++){
    tmp2[k] = 0.0;
    for(int e=0;e<M_;e++)
      polys[k*M_+e] = 0.0;
  }
  
  
  for(int e=0;e<M_;e++){
    
  
  
     for(int j=0;j<SUBDIM_;j++){
       
     for(int k=0;k<SUBDIM_;k++)
       for(int e=0;e<M_;e++)
         polys[k*M_+e] = 0.0;

     
      polys[j*M_+e] = 1.0;
     

     batch_vel_op(polys, tmp2);


     
       for(int i=0;i<SUBDIM_;i++){
        

	 if(e==0){
           if(polys[i*M_+e]!=0)
             std::cout<<polys[i*M_+e]/1.35<<"  ";
           else
	     std::cout<<"   ";
	 }
       }
      if(e==0)
           std::cout<<std::endl;
       
     }
     //      std::cout<<e<<"  "<<diff<<std::endl;
   }
  
}

#endif //TEST_HAMILTONIAN_HPP
