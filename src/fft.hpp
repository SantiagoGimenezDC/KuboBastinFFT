#ifndef FFT_HPP
#define FFT_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<cstring>
#include<complex>
#include<fftw3.h>
#include<omp.h>

#include "static_vars.hpp"
#include "kernel.hpp"
#include "complex_op.hpp"


void KuboBastin_FFTs(type bras[SUBDIM_*M_], type kets[SUBDIM_*M_], type E_points[M_], type integrand[2*M_]){
   
  int size = SUBDIM_;
  const std::complex<double> im(0,1);
  //  VectorXp preFactor(N);
    

  std::complex<type> factors[M_];

  for(int m=0;m<M_;m++)
    factors[m] = (2-(m==0)) * kernel(m) *  std::polar(1.0,M_PI*m/(2.0*M_)) ;
    

    #pragma omp parallel 
    {

    int id,  Nthrds, l_start, l_end;
    id = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end = (id+1) * size / Nthrds;

    double thread_integrand[M_];

    for(int m=0;m<M_;m++)
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
    
    bra   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M_ );    
    bra_d = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M_ );
    
    ket   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M_ );    
    ket_d = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M_ );
    
    

    

    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M_, bra, bra, FFTW_FORWARD, FFTW_ESTIMATE);//bra1
      plan2 = fftw_plan_dft_1d(M_, bra_d, bra_d, FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative
      plan3 = fftw_plan_dft_1d(M_, ket, ket, FFTW_BACKWARD, FFTW_ESTIMATE); //ket1
      plan4 = fftw_plan_dft_1d(M_, ket_d, ket_d, FFTW_BACKWARD, FFTW_ESTIMATE); //ket_derivative
    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M_;m++){
	bra[m][0] = ( factors[m] * bras[l*M_+m]).real();
	bra[m][1] = ( factors[m] * bras[l*M_+m]).imag();
      }
      
      fftw_execute(plan1);
      
      
   
      for(int m=0;m<M_;m++){
	bra_d[m][0] = m * ( factors[m] * bras[l*M_+m] ).real();
        bra_d[m][1] = m * ( factors[m] * bras[l*M_+m] ).imag();
      }
     
      fftw_execute(plan2);





      

      
   
      for(int n=0;n<M_;n++){
	ket[n][0] = ( factors[n] * kets[l*M_+n] ).real();
        ket[n][1] = ( factors[n] * kets[l*M_+n] ).imag(); 
      }
   
      fftw_execute(plan3);
      
   
      for(int n=0;n<M_;n++){
	ket_d[n][0] = n * ( factors[n] * kets[l*M_+n] ).real(); 
        ket_d[n][1] = n * ( factors[n] * kets[l*M_+n] ).imag(); 	
      }
   
      fftw_execute(plan4);

      


      

      
    for(int j=0; j<M_/2; j++ ){  
      type ej = E_points[2*j];
      thread_integrand[2*j] += (
				      (
				        ej                 * (  ket[j][0]   + im * ket[j][1] )  -
				        im * sqrt(1-ej*ej) * (  ket_d[j][0] + im * ket_d[j][1] )
				      ) * bra[j][0]
				      +
				      (
				        ej                 * (  bra[j][0]   + im * bra[j][1]  )  +
				        im * sqrt(1-ej*ej) * (  bra_d[j][0] + im * bra_d[j][1] )
				       ) * ket[j][0]
					
		        ).real();	 
      

      
      ej = E_points[2*j+1];
      thread_integrand[2*j+1] += (
				      (
				       ej                 * (  ket[M_-j-1][0]   - im * ket[M_-j-1][1] )  -
				       im * sqrt(1-ej*ej) * (  ket_d[M_-j-1][0] - im * ket_d[M_-j-1][1] )
				      ) * bra[M_-j-1][0]
				      +
				      (
				       ej                 * (  bra[M_-j-1][0]   - im * bra[M_-j-1][1] )  +
				       im * sqrt(1-ej*ej) * (  bra_d[M_-j-1][0] - im * bra_d[M_-j-1][1] )
				      ) * ket[M_-j-1][0]
			  ).real();
     
      
    }
 

    }

    # pragma omp critical
    {
      for(int m=0;m<M_;m++)
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

    








void batch_fft(type polys[SUBDIM_*M_]){

# pragma omp parallel
    {

    int id,  Nthrds, l_start, l_end, howmany;
    id = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    l_start = id * SUBDIM_ / Nthrds;
    l_end = (id+1) * SUBDIM_ / Nthrds;

    howmany=l_end-l_start;

    
    int n[]={M_};
    
    
   fftw_r2r_kind *r2hckinds = NULL;
   r2hckinds = (fftw_r2r_kind*)malloc(howmany*sizeof(fftw_r2r_kind));
   for (int i=0; i<howmany; ++i)
     r2hckinds[i] = FFTW_REDFT01;
    
    
    
    
    if (id == Nthrds-1) l_end = SUBDIM_;
    
    fftw_plan plan;
    
  


#pragma omp critical
    
    plan = fftw_plan_many_r2r(1, n, howmany,
			    &polys[l_start*M_], n,
			      1, M_,
                             &polys[l_start*M_], n,
			      1, M_,
                             r2hckinds, FFTW_ESTIMATE);
    
    fftw_execute(plan);

  

#pragma omp critical
    {      
        fftw_destroy_plan(plan);
       fftw_free(r2hckinds);
      }
    }
        
  }

#endif //FFT_HPP
