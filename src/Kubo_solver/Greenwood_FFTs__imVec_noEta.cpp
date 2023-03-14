#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"




void Kubo_solver::Greenwood_FFTs__imVec_noEta(std::complex<r_type> **bras, std::complex<r_type> **kets, r_type E_points[], r_type r_data[]){
  const std::complex<double> im(0,1);

  int M  = parameters_.M_,
    size = parameters_.SECTION_SIZE_;

  
 
  r_type *kernel  = new r_type [M],
    *IM_root      = new r_type[M];

  
  for(int m=0;m<M;m++){
    kernel[m]      =  kernel_->term(m,M);
    IM_root[m]     =  sin( acos( E_points[m] )  );
  }

    #pragma omp parallel 
    {

    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    
    r_type *thread_data = new r_type [M];

    for(int m=0;m<M;m++)
      thread_data[m]=0;

    
    if (id == Nthrds-1) l_end = size;
    
    fftw_plan plan1, plan2, plan3, plan4;

    double
      *bra_re = ( double* ) fftw_malloc(sizeof(double) * M ),
      *bra_im = ( double* ) fftw_malloc(sizeof(double) * M ),
      *ket_re = ( double* ) fftw_malloc(sizeof(double) * M ),
      *ket_im = ( double* ) fftw_malloc(sizeof(double) * M );
    
        
# pragma omp critical
    {
      plan1 = fftw_plan_r2r_1d(M, bra_re, bra_re, FFTW_REDFT01, FFTW_ESTIMATE);//bra_re
      plan2 = fftw_plan_r2r_1d(M, bra_im, bra_im, FFTW_REDFT01, FFTW_ESTIMATE); //bra_im
      
      plan3 = fftw_plan_r2r_1d(M, ket_re, ket_re, FFTW_REDFT01, FFTW_ESTIMATE);//bra_re
      plan4 = fftw_plan_r2r_1d(M, ket_im, ket_im, FFTW_REDFT01, FFTW_ESTIMATE); //bra_im
    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){
	bra_re[m] =  kernel[m] * bras[m][l].real();
        bra_im[m] =  kernel[m] * bras[m][l].imag();

	ket_re[m] =  kernel[m] * kets[m][l].real();
        ket_im[m] =  kernel[m] * kets[m][l].imag(); 
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   

      fftw_execute(plan3);
      fftw_execute(plan4);   
      



    for(int j=0; j<M; j++ )
      thread_data[j] += (  (bra_re[j] - im * bra_im[j] )   *   ( ket_re[j] + im *  ket_im[j] )  ).real();
    
    
    }

    # pragma omp critical
    {
      for(int e=0;e<M;e++)
	r_data[e] += 2.0 * thread_data[e] / ( IM_root[e] * IM_root[e] );

      
      delete []thread_data;
      fftw_destroy_plan(plan1);
      fftw_free(bra_re);
      fftw_destroy_plan(plan2);
      fftw_free(bra_im);

      
      fftw_destroy_plan(plan3);
      fftw_free(ket_re);
      fftw_destroy_plan(plan4);
      fftw_free(ket_im);
    }
  }
    
  delete []IM_root;
  delete []kernel;
}

