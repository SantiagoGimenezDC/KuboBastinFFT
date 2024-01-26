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
#include "../Device/Device.hpp"
#include "../Device/Graphene.hpp"

#include "../complex_op.hpp"
#include "Kubo_solver_filtered.hpp"




Kubo_solver_filtered::Kubo_solver_filtered(solver_vars& parameters, Device& device, KB_filter& filter) : parameters_(parameters), device_(device), filter_(filter)
{

  if(parameters_.cap_choice_==0)
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else if(parameters_.cap_choice_==1)
    cap_      = new Effective_Contact(parameters_.E_min_, parameters_.eta_/parameters_.a_);
  else
    cap_      = new Mandelshtam(parameters_.E_min_, parameters_.eta_/parameters_.a_);


  
  if(parameters_.base_choice_==0)
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);
  else if(parameters_.base_choice_==1)
    vec_base_ = new Complex_Phase(device_.parameters(), parameters_.seed_);
  else
    vec_base_ = new Direct(device_.parameters(), parameters_.seed_);

  if(parameters_.kernel_choice_==0)
    kernel_   = new None();
  else if(parameters_.kernel_choice_==1)
    kernel_   = new Jackson();
  else
    kernel_   = new None();
}
/*
void Station(int millisec, std::string msg ){
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<msg;
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl;

}
*/


void Kubo_solver_filtered::compute(){
  
  auto start0 = std::chrono::steady_clock::now();

  
  device_.build_Hamiltonian();
  device_.setup_velOp();
  
  if(parameters_.a_==1.0){
    r_type Emin, Emax;
    device_.minMax_EigenValues(300, Emax,Emin);

    parameters_.a_ = (Emax-Emin)/(2.0-parameters_.edge_);
    parameters_.b_ = -(Emax+Emin)/2.0;

  }
  
  device_.adimensionalize(parameters_.a_, parameters_.b_);



  

  int W      = device_.parameters().W_,
      C      = device_.parameters().C_,
      LE     = device_.parameters().LE_,
      DIM    = device_.parameters().DIM_,
      SUBDIM = device_.parameters().SUBDIM_;    

  int num_parts = parameters_.num_parts_,
    SEC_SIZE    = 0;

  SEC_SIZE = SUBDIM / num_parts;
  parameters_.SECTION_SIZE_ = SEC_SIZE;

  
  r_type a       = parameters_.a_,
         E_min   = parameters_.E_min_,
    //   E_start = parameters_.E_start_,
    //   E_end   = parameters_.E_end_,
         eta     = parameters_.eta_;
  
  int M = parameters_.M_,
    R = parameters_.R_,
    D = parameters_.dis_real_,
    nump = parameters_.num_p_;

  std::string run_dir  = parameters_.run_dir_,
              filename = parameters_.filename_;

  E_min /= a;
  eta   /= a;


  auto start_BT = std::chrono::steady_clock::now();




/*------------Big memory allocation--------------*/
  //Single Shot vectors
  type **bras = new type* [ M ],
       **kets = new type* [ M ];

  for(int m=0;m<M;m++){
    bras[m] = new type [ SEC_SIZE ],
    kets[m] = new type [ SEC_SIZE ];
  }

  
  //Recursion Vectors
  type *vec      = new type [ DIM ],
       *p_vec    = new type [ DIM ],
       *pp_vec   = new type [ DIM ],
       *rand_vec = new type [ DIM ];
  
  //Auxiliary - disorder and CAP vectors
  r_type *dmp_op  = new r_type [ DIM ],
         *dis_vec = new r_type [ SUBDIM ];
/*-----------------------------------------------*/




  
/*---------------Dataset vectors----------------*/
  r_type *r_data     = new r_type [ nump ],
         *final_data = new r_type [ nump ],
         *E_points   = new r_type [ nump ],
         *conv_R     = new r_type [ 2*D*R ];

  r_type *integrand  = new r_type [nump] ;
/*-----------------------------------------------*/  




  
/*----------------Initializations----------------*/
#pragma omp parallel for  
  for(int m=0;m<M;m++)
    for(int n=0;n<SEC_SIZE;n++){    
      bras [m][n] = 0.0;
      kets [m][n] = 0.0;
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
  for(int e=0; e<nump;e++){
    r_data     [e] = 0.0;
    final_data [e] = 0.0;
    integrand  [e] = 0;
    if(!filter_.parameters().filter_ && !filter_.parameters().post_filter_)
      E_points   [e] = cos( M_PI * ( (r_type)e + 0.5) / (r_type)M );
  }
/*-----------------------------------------------*/  
   

  filter_.compute_filter();
  if(filter_.parameters().filter_ || filter_.parameters().post_filter_)
    E_points = filter_.E_points();

  cap_->create_CAP(W, C, LE,  dmp_op);
  device_.damp(dmp_op);
  

  
  auto end_BT = std::chrono::steady_clock::now();
  Station( std::chrono::duration_cast<std::chrono::microseconds>(end_BT - start_BT).count()/1000, "    Bloat time:            ");


  
  for(int d=1; d<=D;d++){

    int total_csrmv = 0,
      total_FFTs    = 0;

    
    device_.Anderson_disorder(dis_vec);
    device_.update_dis(dis_vec, dmp_op);
    
    for(int r=1; r<=R;r++){
       
      auto start_RV = std::chrono::steady_clock::now();
      std::cout<<std::endl<<d*r<<"/"<< D*R<< "-Vector/disorder realization;"<<std::endl;



       vec_base_->generate_vec_im( rand_vec, r);       
       device_.rearrange_initial_vec(rand_vec); //very hacky
  

    
       for(int k=0; k<nump; k++ ){
         r_data    [k] = 0;
         integrand [k] = 0;
       }

       for(int s=0;s<=num_parts;s++){

	 if( s==num_parts && SUBDIM % num_parts==0  )
	   break;

	 if(SUBDIM % num_parts==0)
           std::cout<< "    -Part: "<<s+1<<"/"<<num_parts<<std::endl;
         else
	   std::cout<< "    -Part: "<<s+1<<"/"<<num_parts+1<<std::endl;
	 
         auto csrmv_start = std::chrono::steady_clock::now();
	 for(int k=0;k<DIM;k++){
           vec     [k] = 0.0;
           pp_vec  [k] = rand_vec[k];
           p_vec   [k] = 0.0;
         }    

         polynomial_cycle_ket( kets, vec, p_vec, pp_vec, dmp_op, dis_vec, s);

    
         auto csrmv_end = std::chrono::steady_clock::now();
         Station( std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000, "           Kets cycle time:            ");

	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count()/1000;
    

	 /*---------------Post process filter--------------*/
         if(filter_.parameters().post_filter_)
	   filter_.post_process_filter(kets, SUBDIM);

         /*------------------------------------------------*/

         auto csrmv_start_2 = std::chrono::steady_clock::now();

#pragma omp parallel for
         for(int k=0;k<DIM;k++){
           vec    [k] = 0.0;
           p_vec  [k] = 0.0;
           pp_vec [k] = 0.0;
         }
    
         device_.vel_op( pp_vec, rand_vec );
         polynomial_cycle(  bras, vec, p_vec, pp_vec, dmp_op, dis_vec, s);	

         auto csrmv_end_2 = std::chrono::steady_clock::now();
         Station(std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000, "           Bras cycle time:            ");
  
	 total_csrmv += std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count()/1000;  


	 /*---------------Post process filter--------------*/
         if(filter_.parameters().post_filter_)
	   filter_.post_process_filter(bras, SUBDIM);
	 
         /*------------------------------------------------*/

         auto FFT_start_2 = std::chrono::steady_clock::now();

         //Bastin_FFTs__reVec_noEta(bras,kets, E_points, integrand);    	 
         //Bastin_FFTs__imVec_noEta(bras,kets, E_points, integrand);
         
	 
	 //Greenwood_FFTs__reVec_noEta(bras,kets, E_points, r_data);         
         

         if(filter_.parameters().post_filter_ || filter_.parameters().filter_)
	   Greenwood_FFTs__imVec_eta_adjusted(bras,kets, E_points, r_data);
	 else
	   Greenwood_FFTs__imVec_noEta_adjusted(bras,kets, E_points, r_data);
	 
	 /*
         These 3 are meant to correct the pre factor .As it turns out, for small values of eta<0.1,
	 those corrections are almost invisible and not worthwhile the huge increase in computational
	 cost. The correction is meaningfull at the -1 and 1 edges, however, it is a lot more 
         reasonable to adjuste edge_ variable to deal with those 
	 
	 //Bastin_FFTs__imVec_eta(bras,kets, E_points, integrand);
         //Greenwood_FFTs__reVec_eta(bras,kets, E_points, r_data);
	 
	 
         */
	 
	 auto FFT_end_2 = std::chrono::steady_clock::now();
         Station(std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count()/1000, "           FFT operations time:        ");
    
	 total_FFTs += std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count()/1000;

       }

       std::cout<<std::endl<<"       Total CSRMV time:           "<< total_csrmv<<" (ms)"<<std::endl;
       std::cout<<"       Total FFTs time:            "<< total_FFTs<<" (ms)"<<std::endl;

       
       auto start_pr = std::chrono::steady_clock::now();
    
       //integration(E_points, integrand, r_data);
    
       auto end_pr = std::chrono::steady_clock::now();
       Station(std::chrono::duration_cast<std::chrono::microseconds>(end_pr - start_pr).count()/1000, "       Integration time:           ");
       

       
    
       auto plot_start = std::chrono::steady_clock::now();    

       /*When introducing a const. eta with modified polynomials, the result is equals to that of a
       simulation with regular polynomials and an variable eta_{var}=eta*sin(acos(E)). The following
       heuristical correction greatly improves the result far from the CNP to match that of the
       desired regular polys and const. eta.*/
       if( parameters_.eta_!=0 )
         eta_CAP_correct(E_points, r_data);


       update_data(E_points, integrand, r_data, final_data, conv_R, (d-1)*R+r, run_dir, filename);
       plot_data(run_dir,filename);
    
       auto plot_end = std::chrono::steady_clock::now();
       Station(std::chrono::duration_cast<std::chrono::microseconds>(plot_end - plot_start).count()/1000, "       Plot and update time:       ");


    
       
       auto end_RV = std::chrono::steady_clock::now();    
       Station(std::chrono::duration_cast<std::chrono::milliseconds>(end_RV - start_RV).count(), "       Total RandVec time:         ");

       std::cout<<std::endl;
    }
  }


  
/*------------Delete everything--------------*/
  //Single Shot vectors
  for(int m=0;m<M;m++){
    delete []bras[m];
    delete []kets[m];
  }
  delete []bras;
  delete []kets;

  //Recursion Vectors
  delete []vec;
  delete []p_vec;
  delete []pp_vec;
  delete []rand_vec;
  
  //Auxiliary - disorder and CAP vectors
  delete []dmp_op;
  delete []dis_vec;
/*-----------------------------------------------*/




  
/*---------------Dataset vectors----------------*/
  delete []r_data;
  delete []final_data;
  delete []conv_R;
  delete []integrand;

  if(!filter_.parameters().filter_ && !filter_.parameters().post_filter_)  
    delete []E_points;

  /*-----------------------------------------------*/  

  auto end = std::chrono::steady_clock::now();   
  Station(std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count(), "Total case execution time:             ");
  std::cout<<std::endl;
  
}




void Kubo_solver_filtered::polynomial_cycle(type** polys, type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[], int s){
  
  int M = parameters_.M_,
      num_parts = parameters_.num_parts_;


//=================================KPM Step 0======================================//


  device_.traceover(polys[0], pp_vec, s, num_parts);
  

  
//=================================KPM Step 1======================================//   
    
    
  device_.H_ket ( p_vec, pp_vec, damp_op, dis_vec);

  device_.traceover(polys[1], p_vec, s, num_parts);
    
    

//=================================KPM Steps 2 and on===============================//
    
    for( int m=2; m<M; m++ ){
      device_.update_cheb( vec, p_vec, pp_vec, damp_op, dis_vec);
      device_.traceover(polys[m], vec, s, num_parts);
    }
}


void Kubo_solver_filtered::polynomial_cycle_ket(type** polys, type vec[], type p_vec[], type pp_vec[], r_type damp_op[], r_type dis_vec[], int s){

  int M   = parameters_.M_,
      DIM   = device_.parameters().DIM_,
      num_parts = parameters_.num_parts_;

  type *tmp = new type [DIM];
//=================================KPM Step 0======================================//
  
  device_.vel_op( tmp, pp_vec );
  device_.traceover(polys[0], tmp, s, num_parts);  
  
  
//=================================KPM Step 1======================================//       
    
  device_.H_ket ( p_vec, pp_vec, damp_op, dis_vec);
  device_.vel_op( tmp, p_vec );
  
  device_.traceover(polys[1], tmp, s, num_parts);  
  
    

//=================================KPM Steps 2 and on===============================//
    
  for( int m=2; m<M; m++ ){
    device_.update_cheb( vec, p_vec, pp_vec, damp_op, dis_vec);
    device_.vel_op( tmp, vec );
    device_.traceover(polys[m], tmp, s, num_parts);
  }
  
  delete []tmp;
}



void Kubo_solver_filtered::eta_CAP_correct(r_type E_points[], r_type r_data[]){
  int nump = parameters_.num_p_;
  
  for(int e=0;e<nump;e++)
    r_data[e] *= sin(acos(E_points[e]));
}


void Kubo_solver_filtered::integration(r_type E_points[], r_type integrand[], r_type data[]){

  int M = parameters_.M_,
    nump = parameters_.num_p_;
  r_type edge = parameters_.edge_;
  
#pragma omp parallel for 
  for(int k=0; k<M-int(M*edge/4.0); k++ ){  //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                                       //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                                      //the number of points to be ignored in the future;
    for(int j=k; j<M-int(M*edge/4.0); j++ ){//IMPLICIT PARTITION FUNCTION. Energies are decrescent with e (bottom of the band structure is at e=M);
      r_type ej  = E_points[j],
	ej1      = E_points[j+1],
	de       = ej-ej1,
        integ    = ( integrand[j+1] + integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ;
    }
  }
}






void Kubo_solver_filtered::update_data(r_type E_points[], r_type integrand[], r_type r_data[], r_type final_data[], r_type conv_R[], int r, std::string run_dir, std::string filename){

  int nump = parameters_.num_p_,
    R = parameters_.R_,
    D = parameters_.dis_real_;


  int SUBDIM = device_.parameters().SUBDIM_;    

  r_type a = parameters_.a_,
    b = parameters_.b_,
    sysSubLength = device_.sysSubLength();


  
  r_type omega = SUBDIM/( a * a * sysSubLength * sysSubLength );//Dimensional and normalizing constant
  

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
    dataR<< a * E_points[e] - b<<"  "<< r_data [e] <<std::endl;

  dataR.close();



  
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e=0;e<nump;e++)  
    dataP<< a * E_points[e] - b<<"  "<< final_data [e] <<std::endl;
    //  dataP<<  e <<"  "<< final_data [e] <<std::endl;

  dataP.close();


  
  
  std::ofstream data;
  data.open(run_dir+"conv_R_"+filename);

  for(int r=1;r<D*R;r++)  
    data<< r <<"  "<< conv_R[ 2*(r-1) ]<<"  "<< conv_R[ 2*(r-1) + 1 ] <<std::endl;

  data.close();


  
  
  std::ofstream data2;
  data2.open(run_dir+"integrand_"+filename);

  for(int e=0;e<nump;e++)  
    data2<< a * E_points[e] - b<<"  "<< omega * integrand[e] <<std::endl;

  
  data2.close();
}






void Kubo_solver_filtered::plot_data(std::string run_dir, std::string filename){
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









