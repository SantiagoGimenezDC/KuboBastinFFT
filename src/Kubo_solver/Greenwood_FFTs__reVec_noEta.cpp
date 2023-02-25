#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"


void Kubo_solver::Greenwood_FFTs__reVec_noEta(r_type bras[], r_type kets[], r_type E_points[], r_type final_r_data[]){
  
  int M = parameters_.M_;

  
  int size = parameters_.SECTION_SIZE_;
  const std::complex<double> im(0,1);

    

  r_type IM_root [ M ],
         kernel  [ M ],  
         r_data  [ M ];
  
  for(int m=0;m<M;m++){
    kernel  [ m ]  =  kernel_->term(m,M);
    IM_root [ m ] =  sin( acos( E_points[m] )   );
    r_data  [ m ]  = 0;
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

    double  *bra;
    double  *ket;
    
    
    bra   = ( double* ) fftw_malloc(sizeof(double) * M );    
    ket   = ( double* ) fftw_malloc(sizeof(double) * M );
    
    
# pragma omp critical
    {
      plan1 = fftw_plan_r2r_1d(M, bra, bra, FFTW_REDFT01, FFTW_ESTIMATE);//bra
      plan2 = fftw_plan_r2r_1d(M, ket, ket, FFTW_REDFT01, FFTW_ESTIMATE); //ket
      


    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){
	bra[m] =  kernel[m] * bras[m*size+l];
	ket[m] =  kernel[m] * kets[m*size+l];

      }


      fftw_execute(plan1);
      fftw_execute(plan2);   


      


    for(int m=0; m<M; m++ ) 
      thread_data[m] += bra[m] * ket[m];
    
    
    
    }

    # pragma omp critical
    {
      for(int m=0;m<M;m++)
	r_data[m]+=thread_data[m];
      
      fftw_destroy_plan(plan1);
      fftw_free(bra);
      fftw_destroy_plan(plan2);
      fftw_free(ket);

    }
  }

    
  for(int m=0;m<M;m++){
    r_data[m] *= 2.0 / ( IM_root[m] * IM_root[m] );
    final_r_data[m] += r_data[m];
  }
}
