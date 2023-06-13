#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"


void Kubo_solver::Bastin_FFTs__reVec_noEta(r_type** bras, r_type** kets, r_type E_points[], r_type integrand[]){

  int SUBDIM = device_.parameters().SUBDIM_;    

  
  int M = parameters_.M_;

  
  int size = SUBDIM;
  const std::complex<double> im(0,1);
  //  VectorXp preFactor(N);
    

  std::complex<r_type> factors[M];
  r_type IM_root[M];

         



 
  for(int m=0;m<M;m++){
    factors[m] = (2.0-(m==0)) * kernel_->term(m, M) * std::polar(1.0,M_PI*m/(2.0*M)) ;
    IM_root[m] = sin( acos(E_points[m]) );
  }



  

#pragma omp parallel 
    {

    int id,  Nthrds, l_start, l_end;
    id = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end = (id+1) * size / Nthrds;

    r_type thread_integrand[M];

    for(int m=0;m<M;m++)
      thread_integrand[m]=0;

    
    if (id == Nthrds-1) l_end = size;
    
    fftw_plan plan1, plan2, plan3, plan4;

    fftw_complex  *bra;
    fftw_complex  *bra_d;
    fftw_complex  *ket;
    fftw_complex  *ket_d;
    
    
    bra   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );
    bra_d = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );
    
    ket   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );    
    ket_d = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );


    

    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M, bra, bra,     FFTW_FORWARD, FFTW_ESTIMATE);//bra1      
      plan2 = fftw_plan_dft_1d(M, bra_d, bra_d, FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative

      plan3 = fftw_plan_dft_1d(M, ket, ket,     FFTW_BACKWARD, FFTW_ESTIMATE); //ket1
      plan4 = fftw_plan_dft_1d(M, ket_d, ket_d, FFTW_BACKWARD, FFTW_ESTIMATE); //ket_derivative
    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){
	bra[m][0] = ( conj(factors[m] *  bras[m][l] )  ).real();
	bra[m][1] = ( conj(factors[m] *  bras[m][l] )  ).imag();

	bra_d[m][0] = m * bra[m][0]; 
        bra_d[m][1] = m * bra[m][1];


   

	ket[m][0] = ( factors[m] * kets[m][l] ).real();
        ket[m][1] = ( factors[m] * kets[m][l] ).imag(); 

	ket_d[m][0] = m * ket[m][0];
        ket_d[m][1] = m * ket[m][1];

      }


      fftw_execute(plan1);
      fftw_execute(plan2);   
      fftw_execute(plan3);
      fftw_execute(plan4);

      

      

      //Take only the real. part      
    for(int j=0; j<M/2; j++ ){ //Taking the real part only

      r_type ej = E_points[2*j];

      thread_integrand[2*j] += (
				      (
				         ej           *   ket[j][0]   +
				         IM_root[2*j] *  ket_d[j][1] 
					
				      ) *  bra[j][0]
				      +
				      (
				         ej           *   bra[j][0]   -
				         IM_root[2*j] *  bra_d[j][1] 
					
				      ) * ket[j][0]
					
		               );	 
      

      
      ej = E_points[2*j+1];
      thread_integrand[2*j+1] += (
				      (
				        ej             *   ket[M-j-1][0]  -
				        IM_root[2*j+1] * ket_d[M-j-1][1] 
				       
				       ) * bra[M-j-1][0]
				      +
				      (
				        ej             * bra[M-j-1][0]   +
				        IM_root[2*j+1] * bra_d[M-j-1][1] 
				       
				      ) * ket[M-j-1][0]
			         );
      
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

    
#pragma omp parallel for 
  for(int k=0; k<M; k++ ){ 
    r_type ek  = E_points[k];
    integrand[k] *= 2.0/pow((1.0 - ek  * ek ),2.0);
  }

}
