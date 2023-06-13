#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"




void Kubo_solver::Greenwood_FFTs__imVec_noEta(std::complex<r_type> **bras, std::complex<r_type> **kets, r_type E_points[], r_type r_data[]){
  const std::complex<double> im(0,1);

  int M     = parameters_.M_,
      size  = parameters_.SECTION_SIZE_,
      num_p = parameters_.num_p_;
 
  r_type kernel [ M ],
         pre_factor [ num_p ];


#pragma omp parallel for  
  for(int m=0;m<M;m++)
      kernel[m]      =  kernel_->term(m,M);
  

#pragma omp parallel for  
  for(int k=0;k<num_p;k++)
    pre_factor[k]    =  1.0-E_points[k]*E_points[k];
  


  
#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;
    
    if (id == Nthrds-1) l_end = size;


    
    r_type thread_data [num_p];

    for(int k=0;k<num_p;k++)
      thread_data[k]=0;


    
    fftw_plan plan1, plan2, plan3, plan4;

    double
      *input  = ( double* ) fftw_malloc(sizeof(double) * num_p ),      
      *bra_re = ( double* ) fftw_malloc(sizeof(double) * num_p ),
      *bra_im = ( double* ) fftw_malloc(sizeof(double) * num_p ),
      *ket_re = ( double* ) fftw_malloc(sizeof(double) * num_p ),
      *ket_im = ( double* ) fftw_malloc(sizeof(double) * num_p );

    for(int m=0;m<num_p;m++)
      input[m] = 0;



# pragma omp critical
    {      
      plan1 = fftw_plan_r2r_1d(num_p, input, bra_re, FFTW_REDFT01, FFTW_ESTIMATE);
      plan2 = fftw_plan_r2r_1d(num_p, input, bra_im, FFTW_REDFT01, FFTW_ESTIMATE);
      
      plan3 = fftw_plan_r2r_1d(num_p, input, ket_re, FFTW_REDFT01, FFTW_ESTIMATE);
      plan4 = fftw_plan_r2r_1d(num_p, input, ket_im, FFTW_REDFT01, FFTW_ESTIMATE);
    }

    for(int l=l_start; l<l_end;l++){

 
      for(int m=0;m<M;m++)
	input[m] = kernel[m] * bras[m][l].real();
     
      fftw_execute(plan1);


      for(int m=0;m<M;m++)
        input[m] = kernel[m] * bras[m][l].imag();

      fftw_execute(plan2);




      
      for(int m=0;m<M;m++)
	input[m] = kernel[m] * kets[m][l].real();

      fftw_execute(plan3);

      
      for(int m=0;m<M;m++)
        input[m] = kernel[m] * kets[m][l].imag(); 
      
      fftw_execute(plan4);   




      
      for(int k=0; k<num_p; k++ )
        thread_data[k] += (  (bra_re[k] - im * bra_im[k] )   *   ( ket_re[k] + im *  ket_im[k] )  ).real();
    
    }

    # pragma omp critical
    {
      for(int k=0;k<num_p;k++)
	r_data[k] += 2.0 * thread_data[num_p-k-1] / pre_factor[k] ;

      
      fftw_free(input);
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
}

