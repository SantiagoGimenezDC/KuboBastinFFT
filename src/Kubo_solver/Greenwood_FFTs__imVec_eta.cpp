#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"



void Kubo_solver::Greenwood_FFTs__imVec_eta(std::complex<r_type> **bras, std::complex<r_type> **kets, r_type E_points[], r_type r_data[]){
  const std::complex<double> im(0,1);
  
  int M    = parameters_.M_,
      size = parameters_.SECTION_SIZE_;

  
  std::complex<r_type>
    IM_energies[M],
    *factors = new std::complex<r_type> [M],
    *IM_root = new std::complex<r_type> [M];
  
  r_type *kernel = new r_type [M];
  
  r_type a = parameters_.a_,
       eta = parameters_.eta_/a;


  for(int m=0;m<M;m++){
    kernel[m]      =  kernel_->term(m,M);
    factors[m]     =  (2-(m==0)) * kernel_->term(m, M) * std::polar(1.0,M_PI*m/(2.0*M)) ;
    IM_energies[m] = E_points[m]+im*eta;//*sin(E_points[m]);
    IM_root[m]     =  sqrt(1.0-IM_energies[m]*IM_energies[m]);//sin( acos( E_points[m] ) - im * asinh( -eta )  );
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


    std::complex<r_type>
      *bra = new std::complex<r_type> [M],
      *ket = new std::complex<r_type> [M];
      
    fftw_complex  *bra_re, *bra_im;
    fftw_complex  *ket_re, *ket_im;
    
    
    bra_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex ) * M );
    bra_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex ) * M );    

    ket_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );
    ket_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );
    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M, bra_re, bra_re, FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
      plan2 = fftw_plan_dft_1d(M, bra_im, bra_im, FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative

      plan3 = fftw_plan_dft_1d(M, ket_re, ket_re, FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
      plan4 = fftw_plan_dft_1d(M, ket_im, ket_im, FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){
	bra[m] = bras[m][l];
	bra_re[m][0] =  factors[m].real() * bra[m].real();
        bra_re[m][1] =  factors[m].imag() * bra[m].real();

	bra_im[m][0] =  factors[m].real() * bra[m].imag();
        bra_im[m][1] =  factors[m].imag() * bra[m].imag();

	
	ket[m] = kets[m][l];
	ket_re[m][0] = factors[m].real() * ket[m].real();
        ket_re[m][1] = factors[m].imag() * ket[m].real(); 

	ket_im[m][0] = factors[m].real() * ket[m].imag();
        ket_im[m][1] = factors[m].imag() * ket[m].imag();
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   

      fftw_execute(plan3);
      fftw_execute(plan4);   
      

      for(int j=0; j<M/2; j++){
	bra[2*j]   = ( ( bra_re[j][0]     + im * bra_re[j][1]     ) / IM_root[2*j]   ).real()  -  im * ( ( bra_im[j][0]     + im * bra_im[j][1]     ) / IM_root[2*j]   ).real();
        bra[2*j+1] = ( ( bra_re[M-j-1][0] + im * bra_re[M-j-1][1] ) / IM_root[2*j+1] ).real()  -  im * ( ( bra_im[M-j-1][0] + im * bra_im[M-j-1][1] ) / IM_root[2*j+1] ).real();	

	ket[2*j]   = ( ( ket_re[j][0]     + im * ket_re[j][1]     ) / IM_root[2*j]   ).real()  +  im * ( ( ket_im[j][0]     + im * ket_im[j][1]     ) / IM_root[2*j]   ).real();
        ket[2*j+1] = ( ( ket_re[M-j-1][0] + im * ket_re[M-j-1][1] ) / IM_root[2*j+1] ).real()  +  im * ( ( ket_im[M-j-1][0] + im * ket_im[M-j-1][1] ) / IM_root[2*j+1] ).real();	
	
      }


    for(int m=0; m<M; m++ ) 
      thread_data[m]   +=   2.0 * ( bra[m] * ket[m] ).real();
    
    
    }

    # pragma omp critical
    {
      for(int m=0;m<M;m++)
	r_data[m]+=thread_data[m];


      delete []bra;
      delete []ket;      
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

    /*     
  if(parameters_.eta_!=0)
    for(int m=0;m<M;m++)
      r_data[m] *= sin(acos(E_points[m]));
    */
  delete []factors;
  delete []IM_root;
  delete []kernel;
}

