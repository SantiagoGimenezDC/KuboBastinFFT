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


#include "static_vars.hpp"

#include "vec_base.hpp"
#include "CAP.hpp"
#include "kernel.hpp"
#include "Graphene.hpp"

#include "complex_op.hpp"
#include "KuboBastin_solver.hpp"





KuboBastin_solver::KuboBastin_solver(solver_vars& parameters, Graphene& device) : parameters_(parameters), device_(device)
{
  kernel_= new Jackson();
}


void KuboBastin_solver::compute(){
  
  auto start0 = std::chrono::steady_clock::now();


  int W   = device_.parameters().W_,
      C   = device_.parameters().C_,
      LE   = device_.parameters().LE_,
      DIM = device_.parameters().DIM_,
      SUBDIM = device_.parameters().SUBDIM_;    

  type a = parameters_.a_,
       b = parameters_.b_,
       E_min = parameters_.E_min_,
    //       E_start = parameters_.E_start_,
    //   E_end = parameters_.E_end_,
       eta = parameters_.eta_;
  
  int M = parameters_.M_,
    R = parameters_.R_;

  std::string run_dir = parameters_.run_dir_,
    filename = parameters_.filename_;


  
  E_min/=(double)a;
  // E_start = -1;//(E_start-b)/a;
  // E_end   = 1;//(E_end-b)/a;
  eta    /= a;





/*---------All static memory declaration---------*/
  //Single Shot vectors
  type bras[M * SUBDIM],
       kets[M * SUBDIM];
  
  //Recursion Vectors
  type vec[DIM],
       p_vec[DIM],
       pp_vec[DIM],
       rand_vec[DIM];
  
  //Auxiliary - disorder and CAP vectors
  type dmp_op[DIM];


  //Print data
  type r_data[M],
       final_data[M],
       E_points[M];
/*-----------------------------------------------*/    


  type  integrand[M];




  
  for(int k=0;k<DIM;k++){
    vec[k]      = 0.0;
    p_vec[k]    = 0.0;
    pp_vec[k]   = 0.0;
    rand_vec[k] = 0.0;
    dmp_op[k]  = 1.0;
  }
  
  for(int k=0;k<M*SUBDIM;k++){
    bras[k] = 0.0;
    kets[k] = 0.0;
  }

  for(int k=0; k<M;k++){
    r_data[k]     = 0.0;
    final_data[k] = 0.0;
    integrand[k] = 0;
  }



  for(int e=0;e<M;e++)
    E_points[e] = cos(M_PI*((double)e+0.5)/(double)M);
   


  create_CAP(W, C, LE, eta, E_min, dmp_op);

  

  


  
  for(int r=1; r<=R;r++){
       
    auto start_RV = std::chrono::steady_clock::now();
    std::cout<<std::endl<<r<<"/"<< R<< "-Random vector;"<<std::endl;

    int seed = 1000;

        generate_vec(C, W, LE, rand_vec, seed, r);
    //rand_vec[DIM/2+r-1] = 0;
    //rand_vec[DIM/2+r] = 1.0;

    
    auto csrmv_start = std::chrono::steady_clock::now();
    
    for(int k=0; k<M; k++ ){
      r_data[k]=0;
      integrand[k]=0;
    }

    
    for(int k=0;k<DIM;k++){
      vec[k]     = 0.0;
      pp_vec[k]  = 0.0;
      p_vec[k]   = rand_vec[k];
    }    
    
    polynomial_cycle_ket( kets, vec, p_vec, pp_vec, dmp_op, a, b);
    //    device_.batch_vel_op(M, kets, vec );
    
    auto csrmv_end = std::chrono::steady_clock::now();
    int csrmv_time=std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end - csrmv_start).count();    
    std::cout<<"       Kets cycle time:            "<<csrmv_time/1000<<"ms"<<std::endl;  


    


    


    auto csrmv_start_2 = std::chrono::steady_clock::now();

    for(int k=0;k<DIM;k++){
      vec[k]      = 0.0;
      p_vec[k]    = 0.0;
      pp_vec[k]   = 0.0;
    }
    
    device_.vel_op( &(p_vec[C*W]), &(rand_vec[C*W]) );
    polynomial_cycle(  bras, vec, p_vec, pp_vec, dmp_op, a, b);	

    auto csrmv_end_2 = std::chrono::steady_clock::now();
    int csrmv_time_2=std::chrono::duration_cast<std::chrono::microseconds>(csrmv_end_2 - csrmv_start_2).count();    
    std::cout<<"       Bras cycle time:            "<<csrmv_time_2/1000<<"ms"<<std::endl;  





    
    

    auto FFT_start_2 = std::chrono::steady_clock::now();

    KuboBastin_FFTs(bras,kets, E_points, integrand);
    // KuboGreenwood_FFTs(bras,kets, E_points, r_data);    
    auto FFT_end_2 = std::chrono::steady_clock::now();
    int FFT_time_2=std::chrono::duration_cast<std::chrono::microseconds>(FFT_end_2 - FFT_start_2).count();    
    std::cout<<"       FFT operations time:        "<<FFT_time_2/1000<<"ms"<<std::endl;  
    






    
    
    auto start_pr = std::chrono::steady_clock::now();
    
    integration(E_points, integrand, r_data);
    
    auto end_pr = std::chrono::steady_clock::now();
    int int_time=std::chrono::duration_cast<std::chrono::microseconds>(end_pr - start_pr).count();
    std::cout<<"       Integration time:           "<<int_time/1000<<"ms"<<std::endl;    




    
    auto plot_start = std::chrono::steady_clock::now();    

    update_data(E_points, integrand, r_data, final_data, r, run_dir, filename);
    plot_data(run_dir,filename);
    
    auto plot_end = std::chrono::steady_clock::now();
    int plot_time = std::chrono::duration_cast<std::chrono::microseconds>(plot_end - plot_start).count();    
    std::cout<<"       Plot and update time:       "<<plot_time/1000<<"ms"<<std::endl;  
    



    
      
     auto end_RV = std::chrono::steady_clock::now();    

     int millisec=std::chrono::duration_cast<std::chrono::milliseconds>(end_RV - start_RV).count();
     int sec=millisec/1000;
     int min=sec/60;
     int reSec=sec%60;
     std::cout<<std::endl<<"       Total RandVec time:         ";
     std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl<<std::endl<<std::endl;
  }
  
  auto end = std::chrono::steady_clock::now();   

  int millisec=std::chrono::duration_cast<std::chrono::milliseconds>(end - start0).count();
  int sec=millisec/1000;
  int min=sec/60;
  int reSec=sec%60;
  std::cout<<"  Total case execution time:             ";
  std::cout<<min<<" min, "<<reSec<<" secs;"<<" ("<< millisec<<"ms) "<<std::endl<<std::endl<<std::endl;



  
}




void KuboBastin_solver::polynomial_cycle(type polys[], type vec[], type p_vec[], type pp_vec[], type damp_op[],  type a, type b){

  int M = parameters_.M_;

  
  int W   = device_.parameters().W_,
      C   = device_.parameters().C_,
      SUBDIM = device_.parameters().SUBDIM_;    

//=================================KPM Step 0======================================//

#pragma omp parallel for 
  for(int i=0;i<SUBDIM;i++)
    polys[i] = p_vec[i+C*W];


  

  
//=================================KPM Step 1======================================//   
    
    
    device_.update_cheb ( vec, p_vec, pp_vec, damp_op, 2*a, b);
    
#pragma omp parallel for 
    for(int i=0;i<SUBDIM;i++)
      polys[SUBDIM+i] = vec[i+C*W];
    

    


//=================================KPM Steps 2 and on===============================//
    
    for( int m=2; m<M; m++ ){
      device_.update_cheb( vec, p_vec, pp_vec, damp_op, a, b);
      
#pragma omp parallel for 
      for(int i=0;i<SUBDIM;i++)
        polys[m*SUBDIM+i] = vec[i+C*W];
      
    }
      
  
}


void KuboBastin_solver::polynomial_cycle_ket(type polys[], type vec[], type p_vec[], type pp_vec[], type damp_op[],  type a, type b){

  int M = parameters_.M_;

  
  int W   = device_.parameters().W_,
      C   = device_.parameters().C_,
      SUBDIM = device_.parameters().SUBDIM_;    

  
//=================================KPM Step 0======================================//

  device_.vel_op( &(polys[0]), &(p_vec[C*W]) );


  
//=================================KPM Step 1======================================//   
    
    
    device_.update_cheb ( vec, p_vec, pp_vec, damp_op, 2*a, b);
    device_.vel_op( &(polys[SUBDIM]), &(vec[C*W]) );

    

//=================================KPM Steps 2 and on===============================//
    
    for( int m=2; m<M; m++ ){
      device_.update_cheb( vec, p_vec, pp_vec, damp_op, a, b);
      device_.vel_op( &(polys[m*SUBDIM]), &(vec[C*W]) );
      
    }

  
}




void KuboBastin_solver::integration(type E_points[], type integrand[], type data[]){

  int M = parameters_.M_;
  
#pragma omp parallel for 
  for(int k=0; k<M-M/100; k++ ){ //At the very edges of the energy plot the weight function diverges, hence integration should start a little after and a little before the edge.
                              //The safety factor guarantees that the conductivity is zero way before reaching the edge. In fact, the safety factor should be used to determine
                              //the number of points to be ignored in the future;
    for(int j=k; j>M/100; j-- ){//IMPLICIT PARTITION FUNCTION
      type ej  = E_points[j],
           ej1 = E_points[j-1];
		
      type preFactor  = ( 1.0 / (  (1.0 - ej  * ej   )  ) ),
	   preFactor2 = ( 1.0 / (  (1.0 - ej1 * ej1  )  ) );

      preFactor  *= -8.0 * preFactor  /4.0 ; //Minus sign from vel. op (im*im).
      preFactor2 *= -8.0 * preFactor2 /4.0 ;
    
      type de    = ej1-ej,
           integ = ( preFactor2*integrand[j-1] + preFactor * integrand[j] ) / 2.0;     
      
      data[k] +=  de * integ ;
		 
	 
    }
  }
 
}





void KuboBastin_solver::update_data(type E_points[], type integrand[], type r_data[], type final_data[], int r, std::string run_dir, std::string filename){

  int M = parameters_.M_;

  int LE  = device_.parameters().LE_,
      SUBDIM = device_.parameters().SUBDIM_;    

  type a = parameters_.a_,
    b = parameters_.b_;
    
  type omega = SUBDIM/( a * a * (LE-1) * (LE-1) * (1+sin(M_PI/6)) * (1+sin(M_PI/6))   );//Dimensional and normalizing constant
  

  
  for(int e=0;e<M;e++)
    final_data[e] = ( final_data[e] * (r-1.0) + omega * r_data[e] ) / r;


    
  std::ofstream dataP;
  dataP.open(run_dir+"currentResult_"+filename);

  for(int e=0;e<M;e++)  
    dataP<<a*E_points[e]+b<<"  "<<final_data[e]<<std::endl;


  
  std::ofstream data;
  data.open(run_dir+"integrand_"+filename);

  for(int e=0;e<M;e++)  
    data<<a*E_points[e]+b<<"  "<<omega*integrand[e]<<std::endl;

  
  data.close();

}




void KuboBastin_solver::plot_data(std::string run_dir, std::string filename){
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








void KuboBastin_solver::KuboBastin_FFTs(type bras[], type kets[], type E_points[], type integrand[]){

  int SUBDIM = device_.parameters().SUBDIM_;    

  
  int M = parameters_.M_;

  
  int size = SUBDIM;
  const std::complex<double> im(0,1);
  //  VectorXp preFactor(N);
    

  std::complex<type> factors[M],IM_energies[M], IM_root[M];

  type a = parameters_.a_,
       eta = parameters_.eta_/a;

    
  for(int m=0;m<M;m++){
    factors[m] = (2-(m==0)) * kernel_->term(m, M) *  std::polar(1.0,M_PI*m/(2.0*M)) ;
    IM_energies[m] = E_points[m]-im*asinh(-eta/sqrt(1-E_points[m]*E_points[m]));
    IM_root[m] = sqrt(1.0-IM_energies[m]*IM_energies[m]);
  }

    #pragma omp parallel 
    {

    int id,  Nthrds, l_start, l_end;
    id = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end = (id+1) * size / Nthrds;

    double thread_integrand[M];

    for(int m=0;m<M;m++)
      thread_integrand[m]=0;

    
    if (id == Nthrds-1) l_end = size;
    
    //  std::string filenameStr="Wisdom.txt";
    //char filename[filenameStr.size() + 1];
    //std::strcpy(filename, filenameStr.c_str());
    //int fftw_import_wisdom_from_filename(const char *filename);

    fftw_plan plan1, plan2, plan3, plan4;

    fftw_complex  *bra;//[N_e_max_];
    fftw_complex  *bra_d;//[N_e_max_];    
    fftw_complex  *ket;//[N_e_max_];
    fftw_complex  *ket_d;//[N_e_max_];
    
    bra   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );    
    bra_d = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );
    
    ket   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );    
    ket_d = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );
    
    

    

    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M, bra, bra, FFTW_FORWARD, FFTW_ESTIMATE);//bra1
      plan2 = fftw_plan_dft_1d(M, bra_d, bra_d, FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative
      plan3 = fftw_plan_dft_1d(M, ket, ket, FFTW_BACKWARD, FFTW_ESTIMATE); //ket1
      plan4 = fftw_plan_dft_1d(M, ket_d, ket_d, FFTW_BACKWARD, FFTW_ESTIMATE); //ket_derivative
    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){
	bra[m][0] = ( factors[m] * bras[m*SUBDIM+l]).real();
	bra[m][1] = ( factors[m] * bras[m*SUBDIM+l]).imag();

	bra_d[m][0] = m * bra[m][0]; 
        bra_d[m][1] = m * bra[m][1];


   

	ket[m][0] = ( factors[m] * kets[m*SUBDIM+l] ).real();
        ket[m][1] = ( factors[m] * kets[m*SUBDIM+l] ).imag(); 

	ket_d[m][0] = m * ket[m][0];
        ket_d[m][1] = m * ket[m][1];
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   
      fftw_execute(plan3);
      fftw_execute(plan4);

      


      

      //Take only the real. part      
    for(int j=0; j<M/2; j++ ){ //Taking the real part only

      std::complex<type> ej = IM_energies[2*j];
      thread_integrand[2*j] +=(
				      (
				       ej                 * (  ket[j][0]    )  -
				       ( -IM_root[2*j].imag() * ket_d[j][0] - IM_root[2*j].real() * ket_d[j][1] )
				      ) * bra[j][0]
				      +
				      (
				       ej                 * (  bra[j][0]    )  +
				       ( -IM_root[2*j].imag() * bra_d[j][0] - IM_root[2*j].real() * bra_d[j][1] )
				      ) * ket[j][0]
				  ).real();



      
      ej = IM_energies[2*j+1];
      thread_integrand[2*j+1] += (
				      (
				       ej                 * (  ket[M-j-1][0]    )  -
				       ( -IM_root[2*j+1].imag() * ket_d[M-j-1][0] + IM_root[2*j+1].real() * ket_d[M-j-1][1] )
				      ) * bra[M-j-1][0]
				      +
				      (
				       ej                 * (  bra[M-j-1][0]    )  +
				       ( -IM_root[2*j+1].imag() * bra_d[M-j-1][0] + IM_root[2*j+1].real() * bra_d[M-j-1][1] )
				      ) * ket[M-j-1][0]
				  ).real();
     
      
			  }
    
    
    }

    # pragma omp critical
    {
      for(int m=0;m<M;m++)
	integrand[m]+=thread_integrand[m];
      
      fftw_destroy_plan(plan1);
      fftw_free(bra);
      fftw_destroy_plan(plan2);
      fftw_free(bra_d);
      fftw_destroy_plan(plan3);
      fftw_free(ket);
      fftw_destroy_plan(plan4);
      fftw_free(ket_d);
    }
  }
}


void KuboBastin_solver::KuboGreenwood_FFTs(type bras[], type kets[], type E_points[], type r_data[]){

  int SUBDIM = device_.parameters().SUBDIM_;    

  
  int M = parameters_.M_;

  
  int size = SUBDIM;
  const std::complex<double> im(0,1);
  //  VectorXp preFactor(N);
    

  std::complex<type> factors[M],IM_energies[M], IM_root[M];

  type a = parameters_.a_,
       eta = parameters_.eta_/a;

    
  for(int m=0;m<M;m++){
    factors[m] = (2-(m==0)) * kernel_->term(m, M) *  std::polar(1.0,M_PI*m/(2.0*M)) ;
    IM_energies[m] = E_points[m]+im*asinh(-eta)/sqrt(1-E_points[m]*E_points[m]);
    IM_root[m] = sqrt(1.0-IM_energies[m]*IM_energies[m]);
    IM_root[m]*=IM_root[m]/2.0;
  }

    #pragma omp parallel 
    {

    int id,  Nthrds, l_start, l_end;
    id = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end = (id+1) * size / Nthrds;

    double thread_data[M];

    for(int m=0;m<M;m++)
      thread_data[m]=0;

    
    if (id == Nthrds-1) l_end = size;
    
    fftw_plan plan1, plan2;

    fftw_complex  *bra;
    fftw_complex  *ket;
    
    
    bra   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );        
    ket   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );    


    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M, bra, bra, FFTW_BACKWARD, FFTW_ESTIMATE);//bra1
      plan2 = fftw_plan_dft_1d(M, ket, ket, FFTW_BACKWARD, FFTW_ESTIMATE); //ket1
    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){
	bra[m][0] = ( factors[m] * bras[m*SUBDIM+l]).real();
	bra[m][1] = ( factors[m] * bras[m*SUBDIM+l]).imag();

	ket[m][0] = ( factors[m] * kets[m*SUBDIM+l] ).real();
        ket[m][1] = ( factors[m] * kets[m*SUBDIM+l] ).imag(); 
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   

      


      

      //Take only the real. part      
    for(int j=0; j<M/2; j++ ){ //Taking the real part only
      thread_data[2*j]   +=  bra[j][0]*ket[j][0] / (IM_root[2*j]).real();
      thread_data[2*j+1] +=  bra[M-j-1][0]*ket[M-j-1][0] / (IM_root[2*j+1]).real();
    }
    
    
    }

    # pragma omp critical
    {
      for(int m=0;m<M;m++)
	r_data[m]+=thread_data[m];
      
      fftw_destroy_plan(plan1);
      fftw_free(bra);
      fftw_destroy_plan(plan2);
    }
  }
}





/*   { //Lazy Imaginary case

      std::complex<type> ej = IM_energies[2*j];
      thread_integrand[2*j] += (
				      (
				        ej                 * (  ket[j][0]   + im * ket[j][1] )  -
				        im * IM_root[2*j] * (  ket_d[j][0] + im * ket_d[j][1] )
				      ) * bra[j][0]
				      +
				      (
				        ej                 * (  bra[j][0]   + im * bra[j][1]  )  +
				        im * IM_root[2*j] * (  bra_d[j][0] + im * bra_d[j][1] )
				       ) * ket[j][0]
					
		        ).real();	 
      

      
      ej = IM_energies[2*j+1];
      thread_integrand[2*j+1] += (
				      (
				       ej                 * (  ket[M-j-1][0]   - im * ket[M-j-1][1] )  -
				       im * IM_root[2*j+1] * (  ket_d[M-j-1][0] - im * ket_d[M-j-1][1] )
				      ) * bra[M-j-1][0]
				      +
				      (
				       ej                 * (  bra[M-j-1][0]   - im * bra[M-j-1][1] )  +
				       im * IM_root[2*j+1] * (  bra_d[M-j-1][0] - im * bra_d[M-j-1][1] )
				      ) * ket[M-j-1][0]
			  ).real();
     
      
			  }*/


    
      /*{  //Purely real case
      type ej = E_points[2*j];
      thread_integrand[2*j] += (
				      (
				        ej            *  ket[j][0]      +
				        sqrt(1-ej*ej) *  ket_d[j][1] 
				      ) * bra[j][0]
				      +
				      (
				        ej            *   bra[j][0]   -
				        sqrt(1-ej*ej) *   bra_d[j][1] 
				       ) * ket[j][0]
					
		        );	 
      

      
      ej = E_points[2*j+1];
      thread_integrand[2*j+1] += (
				      (
				       ej            *  ket[M-j-1][0]  -
				       sqrt(1-ej*ej) *  ket_d[M-j-1][1] 
				      ) * bra[M-j-1][0]
				      +
				      (
				       ej            *  bra[M-j-1][0] +
				       sqrt(1-ej*ej) *  bra_d[M-j-1][1] 
				      ) * ket[M-j-1][0]
			         );
     
      
    }*/
 

