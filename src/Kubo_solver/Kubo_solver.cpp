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
#include<cstring>

#include<fftw3.h>


#include "../static_vars.hpp"

#include "../vec_base.hpp"
#include "../CAP.hpp"
#include "../kernel.hpp"
#include "../Graphene.hpp"

#include "../complex_op.hpp"
#include "Kubo_solver.hpp"





Kubo_solver::Kubo_solver(solver_vars& parameters, Graphene& device) : parameters_(parameters), device_(device)
{
  kernel_   = new Jackson();
  vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  cap_      = new CAP();
}

void Station(int millisec, std::string msg ){
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<msg;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl;

}



void Kubo_solver::compute(){
  
  auto start0 = std::chrono::steady_clock::now();


  int W      = device_.parameters().W_,
      C      = device_.parameters().C_,
      LE     = device_.parameters().LE_,
      DIM    = device_.parameters().DIM_,
      SUBDIM = device_.parameters().SUBDIM_;    

  int num_parts = parameters_.num_parts_,
    SEC_SIZE    = parameters_.SECTION_SIZE_;

  
  r_type a       = parameters_.a_,
         E_min   = parameters_.E_min_,
    //   E_start = parameters_.E_start_,
    //   E_end   = parameters_.E_end_,
         eta     = parameters_.eta_;
  
  int M = parameters_.M_,
    R = parameters_.R_,
    D = parameters_.dis_real_;

  std::string run_dir  = parameters_.run_dir_,
              filename = parameters_.filename_;

  E_min /= a;
  eta   /= a;


  auto start_BT = std::chrono::steady_clock::now();


/*------------Big memory allocation--------------*/
  //Single Shot vectors
  type bras [ M * SEC_SIZE ],
       kets [ M * SEC_SIZE ];
  
  //Recursion Vectors
  type vec      [ DIM ],
       p_vec    [ DIM ],
       pp_vec   [ DIM ],
       rand_vec [ DIM ];
  
  //Auxiliary - disorder and CAP vectors
  r_type dmp_op  [ DIM ],
         dis_vec [ SUBDIM ];
/*-----------------------------------------------*/




  
/*---------------Dataset vectors----------------*/
  r_type r_data     [ M ],
         final_data [ M ],
         E_points   [ M ],
         conv_R     [ 2*D*R ];

  r_type integrand[M];
/*-----------------------------------------------*/  




  
/*----------------Initializations----------------*/
#pragma omp parallel for  
  for(int k=0;k<M*SEC_SIZE;k++){
    bras [k] = 0.0;
    kets [k] = 0.0;
  }
  
#pragma omp parallel for  
  for(int k=0;k<DIM;k++){
    vec      [k] = 0.0;
    p_vec    [k] = 0.0;
    pp_vec   [k] = 0.0;
    rand_vec [k] = 0.0;
    dmp_op   [k] = 1.0;
  }

#pragma omp parallel for
  for(int r=0;r<D*R;r++)
    conv_R [r] = 0.0;
    
#pragma omp parallel for
  for(int e=0; e<M;e++){
    r_data     [e] = 0.0;
    final_data [e] = 0.0;
    integrand  [e] = 0;
    E_points   [e] = cos( M_PI * ( (r_type)e + 0.5) / (r_type)M );
  }
/*-----------------------------------------------*/  
   


  create_CAP(W, C, LE, eta, E_min, dmp_op);
  //eff_contact(W, C, LE, eta, dmp_op);
  

  
  auto end_BT = std::chrono::steady_clock::now();
  Station( std::chrono::duration_cast<std::chrono::microseconds>(end_BT - start_BT).count()/1000, "    Bloat time:            ");


  
  for(int d=1; d<=D;d++){

    int total_csrmv = 0,
      total_FFTs    = 0;

    
    device_.Anderson_disorder(dis_vec);

    
    for(int r=1; r<=R;r++){
       
      auto start_RV = std::chrono::steady_clock::now();
      std::cout<<std::endl<<d*r<<"/"<< D*R<< "-Vector/disorder realization;"<<std::endl;



       vec_base_->generate_vec_re( rand_vec, r);
       //generate_vec_im(C, W, LE,  rand_vec, this->parameters().seed_, r);



  

    
       for(int k=0; k<M; k++ ){
         r_data    [k] = 0;
         integrand [k] = 0;
       }

       for(int s=0;s<num_parts;s++){

	 
         std::cout<< "    -Part: "<<s+1<<"/"<<num_parts<<std::endl;
	 
         auto csrmv_start = std::chrono::steady_clock::now();
	 for(int k=0;k<DIM;k++){
           vec     [k] = 0.0;
           pp_vec  [k] = 0.0;
           p_vec   [k] = rand_vec[k];
         }    

         polynomial_cycle_ket( kets, vec, p_vec, pp_vec, dmp_op, dis_vec, s);

    
         auto csrmv_end = std::chrono::steady_clock::now();
         Station( std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000, "           Kets cycle time:            ");

	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000;
    


    


         auto csrmv_start_2 = std::chrono::steady_clock::now();

#pragma omp parallel for
         for(int k=0;k<DIM;k++){
           vec    [k] = 0.0;
           p_vec  [k] = 0.0;
           pp_vec [k] = 0.0;
         }
    
         device_.vel_op( &(p_vec[C*W]), &(rand_vec[C*W]) );
         polynomial_cycle(  bras, vec, p_vec, pp_vec, dmp_op, dis_vec, s);	

         auto csrmv_end_2 = std::chrono::steady_clock::now();
         Station(std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000, "           Bras cycle time:            ");
  
	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000;  


 

         auto FFT_start_2 = std::chrono::steady_clock::now();

         //Bastin_FFTs__imVec_noEta_opt(bras,kets, E_points, integrand);
         //Greenwood_FFTs__imVec_noEta(bras,kets, E_points, r_data);
	 Greenwood_FFTs__reVec_noEta(bras,kets, E_points, r_data);
         //Greenwood_FFTs__reVec_eta(bras,kets, E_points, r_data);
         //Bastin_FFTs__reVec_noEta(bras,kets, E_points, integrand);    
         auto FFT_end_2 = std::chrono::steady_clock::now();
         Station(std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count()/1000, "           FFT operations time:        ");
    
	 total_FFTs += std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count()/1000;

       }

       std::cout<<std::endl<<"       Total CSRMV time:           "<< total_csrmv<<" (ms)"<<std::endl;
       std::cout<<"       Total FFTs time:            "<< total_FFTs<<" (ms)"<<std::endl;

       /*
       auto start_pr = std::chrono::steady_clock::now();
    
       integration(E_points, integrand, r_data);
    
       auto end_pr = std::chrono::steady_clock::now();
       Station(std::chrono::duration_cast<std::chrono::microseconds>(end_pr - start_pr).count()/1000, "       Integration time:           ");
       */

       
    
       auto plot_start = std::chrono::steady_clock::now();    

       update_data(E_points, integrand, r_data, final_data, conv_R, (d-1)*R+r, run_dir, filename);
       plot_data(run_dir,filename);
    
       auto plot_end = std::chrono::steady_clock::now();
       Station(std::chrono::duration_cast<std::chrono::microseconds>(plot_end - plot_start).count()/1000, "       Plot and update time:       ");


    
       
       auto end_RV = std::chrono::steady_clock::now();    
       Station(std::chrono::duration_cast<std::chrono::milliseconds>(end_RV - start_RV).count(), "       Total RandVec time:         ");

       std::cout<<std::endl;
    }
  }
  
  auto end = std::chrono::steady_clock::now();   
  Station(std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count(), "Total case execution time:             ");
  std::cout<<std::endl;
  
}







void Kubo_solver::polynomial_cycle(type polys[], type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[], int s){
  
  int M = parameters_.M_;

  int W   = device_.parameters().W_,
      C   = device_.parameters().C_,
    SEC_SIZE = parameters_.SECTION_SIZE_;

  r_type a = parameters_.a_,
    b = parameters_.b_;
//=================================KPM Step 0======================================//

#pragma omp parallel for 
  for(int i=0;i<SEC_SIZE;i++)
    polys[ i ] = p_vec[ s*SEC_SIZE + i+C*W];


  

  
//=================================KPM Step 1======================================//   
    
    
  device_.update_cheb ( vec, p_vec, pp_vec, damp_op, dis_vec, 2*a, b);
    
#pragma omp parallel for 
    for(int i=0;i<SEC_SIZE;i++)
      polys[ SEC_SIZE + i] = vec[s*SEC_SIZE + i+C*W];
    

    


//=================================KPM Steps 2 and on===============================//
    
    for( int m=2; m<M; m++ ){
      device_.update_cheb( vec, p_vec, pp_vec, damp_op, dis_vec,  a, b);
      
#pragma omp parallel for 
      for(int i=0;i<SEC_SIZE;i++)
        polys[ m*SEC_SIZE + i ] = vec[s*SEC_SIZE + i+C*W];
      
    }
}


void Kubo_solver::polynomial_cycle_ket(type polys[], type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[], int s){

  int M   = parameters_.M_,
      W   = device_.parameters().W_,
      C   = device_.parameters().C_,
      SUBDIM   = device_.parameters().SUBDIM_,
      SEC_SIZE = parameters_.SECTION_SIZE_;

  r_type a = parameters_.a_,
    b = parameters_.b_;

  type tmp[SUBDIM];
//=================================KPM Step 0======================================//

  device_.vel_op( tmp, &(p_vec[C*W]) );
  
#pragma omp parallel for 
  for(int i=0;i<SEC_SIZE;i++)
    polys[ i ] = tmp[ s*SEC_SIZE + i];


  
//=================================KPM Step 1======================================//       
    
  device_.update_cheb ( vec, p_vec, pp_vec, damp_op, dis_vec, 2*a, b);
  device_.vel_op( tmp, &(vec[C*W]) );

#pragma omp parallel for 
  for(int i=0;i<SEC_SIZE;i++)
    polys[ SEC_SIZE + i ] = tmp[ s*SEC_SIZE + i];
  
    

//=================================KPM Steps 2 and on===============================//
    
  for( int m=2; m<M; m++ ){
    device_.update_cheb( vec, p_vec, pp_vec, damp_op, dis_vec,  a, b);
    device_.vel_op( tmp, &(vec[C*W]) );      


#pragma omp parallel for 
  for(int i=0;i<SEC_SIZE;i++)
    polys[ m * SEC_SIZE + i ] = tmp[ s*SEC_SIZE + i];
  
  }
}




void Kubo_solver::integration(r_type E_points[], r_type integrand[], r_type data[]){

  int M = parameters_.M_;
  
#pragma omp parallel for 
  for(int k=0; k<M-M/100; k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                  //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                  //the number of points to be ignored in the future;
    for(int j=k; j<M-M/100; j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      r_type ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ;
    }
  } 
}






void Kubo_solver::update_data(r_type E_points[], r_type integrand[], r_type r_data[], r_type final_data[], r_type conv_R[], int r, std::string run_dir, std::string filename){

  int nump = parameters_.M_,
    R = parameters_.R_,
    D = parameters_.dis_real_;
  
  int LE   = device_.parameters().LE_,
    SUBDIM = device_.parameters().SUBDIM_;    

  r_type a = parameters_.a_,
         b = parameters_.b_;
    
  r_type omega = SUBDIM/( a * a * (LE-1) * (LE-1) * (1+sin(M_PI/6)) * (1+sin(M_PI/6))   );//Dimensional and normalizing constant
  

  r_type tmp, max=0, av=0;
  
  for(int e=0;e<nump;e++){

    tmp = final_data[e];
    final_data[e] = ( final_data [e] * (r-1.0) + omega * r_data[e] ) / r;

    if(r>1){

      tmp = std::abs( ( final_data [e] - tmp ) / tmp) ;
      if(tmp>max)
        max = tmp;

      av += tmp / nump ;
    }
  }

  if(r>1){
    conv_R[ 2 * (r-1) ]   = max;
    conv_R[ 2 * (r-1)+1 ] = av;
  }

  std::ofstream dataR;
  dataR.open(run_dir+"vecs/r"+std::to_string(r)+"_"+filename);

  for(int e=0;e<nump;e++)  
    dataR<< a * E_points[e] + b<<"  "<< r_data [e] <<std::endl;

  dataR.close();



  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e=0;e<nump;e++)  
    dataP<< a * E_points[e] + b<<"  "<< final_data [e] <<std::endl;

  dataP.close();


  
  
  std::ofstream data;
  data.open(run_dir+"conv_R_"+filename);

  for(int r=1;r<D*R;r++)  
    data<< r <<"  "<< conv_R[ 2*(r-1) ]<<"  "<< conv_R[ 2*(r-1) + 1 ] <<std::endl;

  data.close();


  
  
  std::ofstream data2;
  data2.open(run_dir+"integrand_"+filename);

  for(int e=0;e<nump;e++)  
    data2<< a * E_points[e] + b<<"  "<< omega * integrand[e] <<std::endl;

  
  data2.close();
}






void Kubo_solver::plot_data(std::string run_dir, std::string filename){
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
      if(system(exeChar)){};


}





