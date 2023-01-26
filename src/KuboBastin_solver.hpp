#ifndef KUBO_BASTIN_SOLVER_HPP
#define KUBO_BASTIN_SOLVER_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<chrono>
#include<ctime>
#include<omp.h>
#include<iomanip>
#include<cstdlib>
#include<thread>
#include<complex>



#include "static_vars.hpp"

#include "vec_base.hpp"
#include "CAP.hpp"
#include "kernel.hpp"
#include "polynomial_cycle.hpp"
#include "fft.hpp"

#include "complex_op.hpp"

struct solver_vars{
  
  type a_ ,b_, E_min_, eta_, E_start_, E_end_, edge_;//m_str, rsh_str, anderson_str;
  int R_;
  std::string filename_, run_dir_;
  
};


void integration(type*, type*, type* );
void update_data(type*, type*, type*, int , type , std::string, std::string );
void plot_data(std::string, std::string );


void KuboBastin_solver(solver_vars& s_vars){
  
  auto start0 = std::chrono::steady_clock::now();

  type a = s_vars.a_,
       b = s_vars.b_,
       E_min = s_vars.E_min_,
       E_start = s_vars.E_start_,
       E_end = s_vars.E_end_,
       eta = s_vars.eta_;
  
  int R = s_vars.R_;

  std::string run_dir = s_vars.run_dir_,
    filename = s_vars.filename_;


  
  E_min/=(double)a;
  E_start = -1;//(E_start-b)/a;
  E_end   = 1;//(E_end-b)/a;
  eta    /= a;


  


/*---------All static memory declaration---------*/
  //Single Shot vectors
  type bras[M_ * SUBDIM_],
       kets[M_ * SUBDIM_];
  
  //Recursion Vectors
  type vec[DIM_],
       p_vec[DIM_],
       pp_vec[DIM_],
       rand_vec[DIM_];
  
  //Auxiliary - disorder and CAP vectors
  type dmp_op[DIM_];


  //Print data
  type r_data[M_],
       final_data[M_],
       E_points[M_];
/*-----------------------------------------------*/    


  type  integrand[M_];




  
  for(int k=0;k<DIM_;k++){
    vec[k]      = 0.0;
    p_vec[k]    = 0.0;
    pp_vec[k]   = 0.0;
    rand_vec[k] = 0.0;
    dmp_op[k]  = 1.0;
  }
  
  for(int k=0;k<M_*SUBDIM_;k++){
    bras[k] = 0.0;
    kets[k] = 0.0;
  }

  for(int k=0; k<M_;k++){
    r_data[k]     = 0.0;
    final_data[k] = 0.0;
    integrand[k] = 0;
  }



  for(int e=0;e<M_;e++)
    E_points[e] = cos(M_PI*((double)e+0.5)/(double)M_);
   

  set_CAP( E_min, dmp_op);

  
  //   print_hamiltonian();  

  


  
  for(int r=1; r<=R;r++){
       
    auto start_RV = std::chrono::steady_clock::now();
    std::cout<<std::endl<<r<<"/"<< R<< "-Random vector;"<<std::endl;

    rand_vec[DIM_/2+r-1] = 0.0;
    rand_vec[DIM_/2+r] = 1.0;



    
    auto csrmv_start = std::chrono::steady_clock::now();
    
    for(int k=0; k<M_; k++ ){
      r_data[k]=0;
      integrand[k]=0;
    }

    
    for(int k=0;k<DIM_;k++){
      vec[k]     = 0.0;
      pp_vec[k]  = 0.0;
      p_vec[k]   = rand_vec[k];
    }    
    
    polynomial_cycle( kets, vec, p_vec, pp_vec, dmp_op, a, b);
    batch_vel_op( kets, vec );
    
    auto csrmv_end = std::chrono::steady_clock::now();
    int csrmv_time=std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count();    
    std::cout<<"       Kets cycle time:            "<<csrmv_time/1000<<"ms"<<std::endl;  


    


    


    auto csrmv_start_2 = std::chrono::steady_clock::now();

    for(int k=0;k<DIM_;k++){
      vec[k]      = 0.0;
      p_vec[k]    = 0.0;
      pp_vec[k]   = 0.0;
    }
    
    vel_op( &(p_vec[C_*W_]), &(rand_vec[C_*W_]) );
    polynomial_cycle(  bras, vec, p_vec, pp_vec, dmp_op, a, b);	

    auto csrmv_end_2 = std::chrono::steady_clock::now();
    int csrmv_time_2=std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count();    
    std::cout<<"       Bras cycle time:            "<<csrmv_time_2/1000<<"ms"<<std::endl;  





    
    

    auto FFT_start_2 = std::chrono::steady_clock::now();

    KuboBastin_FFTs(bras,kets, E_points, integrand);
    
    auto FFT_end_2 = std::chrono::steady_clock::now();
    int FFT_time_2=std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count();    
    std::cout<<"       FFT operations time:        "<<FFT_time_2/1000<<"ms"<<std::endl;  
    






    
    
    auto start_pr = std::chrono::steady_clock::now();
    
    integration(E_points, integrand, r_data);
    
    auto end_pr = std::chrono::steady_clock::now();
    int int_time=std::chrono::duration_cast<std::chrono::microseconds>(end_pr - start_pr).count();
    std::cout<<"       Integration time:           "<<int_time/1000<<"ms"<<std::endl;    



    
    update_data(E_points, r_data, final_data, r, a, run_dir, filename);
    plot_data(run_dir,filename);
    



    
      
     auto end_RV = std::chrono::steady_clock::now();    

     int millisec=std::chrono::duration_cast<std::chrono::milliseconds>(end_RV - start_RV).count();
     int sec=millisec/1000;
     int min=sec/60;
     int reSec=sec%60;
     std::cout<<std::endl<<"       Total RandVec time:         ";
     std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl<<std::endl<<std::endl;
  }
  
  auto end = std::chrono::steady_clock::now();   
  int time= std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count();
  std::cout<<"  Total case execution time:             "<<time/1000<<"s"<<std::endl;    


  
}


void integration(type E_points[M_], type integrand[M_], type data[M_]){
//----------Sample data
    
#pragma omp parallel for 
for(int k=0; k<M_; k++ ){
    for(int j=k; j>0; j-- ){//IMPLICIT PARTITION FUNCTION
      type ej  = E_points[j],
           ej1 = E_points[j-1];
		
      type preFactor  = ( 1.0 / (  (1.0-pow(ej, 2.0))  ) ),
	   preFactor2 = ( 1.0 / (  (1.0-pow(ej1,2.0))  ) );

      preFactor  *= -8.0 * preFactor  / 4.0 /2.0; //Minus sign from vel. op (im*im). Arbitrary division by 2! idk where it comes from
      preFactor2 *= -8.0 * preFactor2 / 4.0 /2.0;
    
      type de    = ej1-ej,
           integ = ( preFactor2*integrand[j-1] + preFactor * integrand[j] ) / 2.0;     
      
      data[k] += ( de * integ );
		 
	 
    }
}
 
}





void update_data(type E_points[M_], type r_data[M_], type final_data[M_], int r, type a, std::string run_dir, std::string filename){

  
  type omega =  SUBDIM_/( a * a * LE_ * LE_  );//Dimensional and normalizing constant
  

  
  for(int e=0;e<M_;e++)
    final_data[e] = ( final_data[e] * (r-1.0) + omega * r_data[e] ) / r;


    
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e=0;e<M_;e++)  
    dataP<<a*E_points[e]<<"  "<<final_data[e]<<std::endl;

  
  dataP.close();

}



void plot_data(std::string run_dir, std::string filename){
        //VIEW commands

     std::string exestring=
         "gnuplot<<EOF                                               \n"
         "set encoding utf8                                          \n"
         "set terminal pngcairo enhanced                             \n"

         "unset key  \n"

         "set output '"+run_dir+filename+".png'                \n"

         "set xlabel 'E[eV]'                                               \n"
         "set ylabel  'G [2e^2/h]'                                           \n"
         
        "plot '"+run_dir+"currentResult_"+filename+"' using 1:2 w p ls 7 ps 0.25 lc 2;  \n"
         "EOF";
     
      char exeChar[exestring.size() + 1];
      strcpy(exeChar, exestring.c_str());    
      system(exeChar);


}





#endif //KUBO_BASTIN_SOLVER_HPP
