#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"


void Kubo_solver::Greenwood_FFTs__reVec_eta(r_type bras[], r_type kets[], r_type E_points[], r_type r_data[]){

  int SUBDIM = device_.parameters().SUBDIM_;    

  
  int M = parameters_.M_;

  
  int size = SUBDIM;
  const std::complex<double> im(0,1);
  //  VectorXp preFactor(N);
    

  std::complex<r_type> factors[M],IM_root[M];

  r_type a = parameters_.a_,
       eta = parameters_.eta_/a;

  
  for(int m=0;m<M;m++){
    factors[m]     =  (2-(m==0)) * kernel_->term(m, M) *  std::polar(1.0,M_PI*m/(2.0*M)) ;
    IM_root[m]     =  sin( acos( E_points[m] ) - im * asinh( eta )  );
    //IM_root[m]     *= IM_root[m]/2.0;
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
	bra[m][0] =  ( factors[m] * bras[m*SUBDIM+l] ).real();
        bra[m][1] =  ( factors[m] * bras[m*SUBDIM+l] ).imag();

	ket[m][0] = ( factors[m] * kets[m*SUBDIM+l] ).real();
        ket[m][1] = ( factors[m] * kets[m*SUBDIM+l] ).imag(); 
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   

      


      


    for(int j=0; j<M/2; j++ ){ 
      thread_data[2*j]   +=   2.0 * (  (bra[j][0] + im * bra[j][1] ) / IM_root[2*j] ).real()            *  (  ( ket[j][0]     + im *  ket[j][1] ) / IM_root[2*j] ).real();
      
      thread_data[2*j+1] +=   2.0 * (  (bra[M-j-1][0] - im * bra[M-j-1][1] ) / IM_root[2*j+1] ).real()  *  (  ( ket[M-j-1][0] - im *  ket[M-j-1][1] ) / IM_root[2*j+1] ).real();

    }
    
    
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
}

