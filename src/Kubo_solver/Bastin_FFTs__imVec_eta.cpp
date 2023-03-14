#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"


void Kubo_solver::Bastin_FFTs__imVec_eta(std::complex<r_type>** bras, std::complex<r_type>** kets, r_type E_points[], r_type final_integrand[]){

  const std::complex<double> im(0,1);
  
  int M  = parameters_.M_,
    size = parameters_.SECTION_SIZE_;
  
    
  std::complex<r_type>
    *pre_factors = new std::complex<r_type> [M],    
    *factors     = new std::complex<r_type> [M],
    *IM_energies = new std::complex<r_type> [M],
    *IM_root     = new std::complex<r_type> [M];
  


  
  r_type a = parameters_.a_,
         eta = parameters_.eta_/a;
  


 
  for(int m=0;m<M;m++){
    IM_energies [m] = E_points[m] + im * eta * sin(acos(E_points[m]));  
    IM_root     [m] = sqrt(1.0 - IM_energies[m] * IM_energies[m] );//sin( acos(E_points[m]) );

    factors     [m] = (2.0-(m==0)) * kernel_->term(m, M) * std::polar(1.0,M_PI*m/(2.0*M)) ;    
    pre_factors [m] = 2.0/pow((1.0 - IM_energies[m]  * IM_energies[m] ),2.0);
  }


  /*  
  r_type *integrand = new r_type [M];

  for(int m=0;m<M;m++)
    integrand[m]=0;
  */

#pragma omp parallel 
    {

    int id,  Nthrds, l_start, l_end;
    id = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end = (id+1) * size / Nthrds;
    
    if (id == Nthrds-1) l_end = size;


    
    //8 planos+ 14 vetores [M] por thread. Pode ser reduzido a 2 planos e uns 4 vetores?
    
    fftw_plan plan1, plan2, plan3, plan4,
              plan5, plan6, plan7, plan8;

    std::complex<r_type>
      *bra_Green = new std::complex<r_type> [M],
      *bra_Delta = new std::complex<r_type> [M],
      *bra_Dfull = new std::complex<r_type> [M],
      
      *ket_Green = new std::complex<r_type> [M],
      *ket_Delta = new std::complex<r_type> [M],
      *ket_Dfull = new std::complex<r_type> [M];   

    
    fftw_complex   
      *bra_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *bra_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),

      *bra_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *bra_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),


      *ket_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *ket_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),

      *ket_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *ket_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );


    
    r_type *thread_integrand = new r_type [M];

    for(int m=0;m<M;m++)
      thread_integrand[m]=0;

    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M, bra_re,   bra_re,    FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative
      plan2 = fftw_plan_dft_1d(M, bra_im,   bra_im,    FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative      
      plan3 = fftw_plan_dft_1d(M, bra_D_re, bra_D_re,  FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative
      plan4 = fftw_plan_dft_1d(M, bra_D_im, bra_D_im,  FFTW_FORWARD, FFTW_ESTIMATE); //bra_derivative

      plan5 = fftw_plan_dft_1d(M, ket_re,   ket_re,    FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
      plan6 = fftw_plan_dft_1d(M, ket_im,   ket_im,    FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative      
      plan7 = fftw_plan_dft_1d(M, ket_D_re, ket_D_re,  FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
      plan8 = fftw_plan_dft_1d(M, ket_D_im, ket_D_im,  FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative



    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){

	bra_Green[m] = bras[m][l]; //This single access row-wise access of a col-major matrix is the longest operation in this loop!!
	
	bra_re[m][0] = ( conj(factors[m]) * bra_Green[m].real() ).real(); 
	bra_re[m][1] = ( conj(factors[m]) * bra_Green[m].real() ).imag(); 


	bra_im[m][0] = ( conj(factors[m]) * bra_Green[m].imag() ).real(); 
	bra_im[m][1] = ( conj(factors[m]) * bra_Green[m].imag() ).imag(); 


	bra_D_re[m][0] = m * bra_re[m][0]; 
	bra_D_re[m][1] = m * bra_re[m][1]; 


	bra_D_im[m][0] = m * bra_im[m][0]; 
	bra_D_im[m][1] = m * bra_im[m][1]; 




	ket_Green[m] = kets[m][l]; 	

	ket_re[m][0] = ( factors[m] * (ket_Green[m].real()) ).real(); 
	ket_re[m][1] = ( factors[m] * (ket_Green[m].real()) ).imag(); 


	ket_im[m][0] = ( factors[m] * (ket_Green[m].imag()) ).real(); 
	ket_im[m][1] = ( factors[m] * (ket_Green[m].imag()) ).imag(); 


	ket_D_re[m][0] = m * ket_re[m][0]; 
	ket_D_re[m][1] = m * ket_re[m][1]; 


	ket_D_im[m][0] = m * ket_im[m][0]; 
	ket_D_im[m][1] = m * ket_im[m][1]; 
	
      }


      fftw_execute(plan1);
      fftw_execute(plan2);   
      fftw_execute(plan3);
      fftw_execute(plan4);

      fftw_execute(plan5);
      fftw_execute(plan6);
      fftw_execute(plan7);
      fftw_execute(plan8);
      

      for(int j=0; j<M/2; j++){
	
	bra_Delta[2*j] = bra_re  [j][0]                       - im *   bra_im  [j][0];	
	bra_Green[2*j] = bra_re  [j][0] + im * bra_re  [j][1] - im * ( bra_im  [j][0] + im * bra_im  [j][1] );//BRA is conjugated for the dot product
	bra_Dfull[2*j] = bra_D_re[j][0] + im * bra_D_re[j][1] - im * ( bra_D_im[j][0] + im * bra_D_im[j][1] );

	ket_Delta[2*j] = ket_re  [j][0]                       + im *   ket_im  [j][0];	
	ket_Green[2*j] = ket_re  [j][0] + im * ket_re  [j][1] + im * ( ket_im  [j][0] + im * ket_im  [j][1] );
	ket_Dfull[2*j] = ket_D_re[j][0] + im * ket_D_re[j][1] + im * ( ket_D_im[j][0] + im * ket_D_im[j][1] );


	
	bra_Delta[2*j+1] = bra_re  [M-j-1][0]                           - im *   bra_im  [M-j-1][0];	
	bra_Green[2*j+1] = bra_re  [M-j-1][0] - im * bra_re  [M-j-1][1] - im * ( bra_im  [M-j-1][0] - im * bra_im  [M-j-1][1] );//FFT algo conversion forces conjugation in re and im parts
	bra_Dfull[2*j+1] = bra_D_re[M-j-1][0] - im * bra_D_re[M-j-1][1] - im * ( bra_D_im[M-j-1][0] - im * bra_D_im[M-j-1][1] );

	ket_Delta[2*j+1] = ket_re  [M-j-1][0]                           + im * ( ket_im  [M-j-1][0]  );	
	ket_Green[2*j+1] = ket_re  [M-j-1][0] - im * ket_re  [M-j-1][1] + im * ( ket_im  [M-j-1][0] - im * ket_im  [M-j-1][1] );//FFT algo conversion forces conjugation in re and im parts
	ket_Dfull[2*j+1] = ket_D_re[M-j-1][0] - im * ket_D_re[M-j-1][1] + im * ( ket_D_im[M-j-1][0] - im * ket_D_im[M-j-1][1] );
      }
      




    for(int m=0; m<M; m++ )
      thread_integrand[m] += (
			      pre_factors[m] * 
			       (
			          bra_Delta[m] * ( IM_energies[m] * ket_Green[m] - im * IM_root[m] * ket_Dfull[m] ) +
			          ket_Delta[m] * ( IM_energies[m] * bra_Green[m] + im * IM_root[m] * bra_Dfull[m] )   
 			       )
	                     ).real();
    
    
    }
    # pragma omp critical
    {
      for(int m=0;m<M;m++)
	final_integrand[m] += thread_integrand[m];


      
      fftw_destroy_plan(plan1);
      fftw_free(bra_re);
      fftw_destroy_plan(plan2);
      fftw_free(bra_im);
      
      fftw_destroy_plan(plan3);
      fftw_free(ket_re);
      fftw_destroy_plan(plan4);
      fftw_free(ket_im); 

      fftw_destroy_plan(plan5);
      fftw_free(bra_D_re);
      fftw_destroy_plan(plan6);
      fftw_free(bra_D_im); 

      
      fftw_destroy_plan(plan7);
      fftw_free(ket_D_re);
      fftw_destroy_plan(plan8);
      fftw_free(ket_D_im); 
      

      delete []bra_Green;
      delete []bra_Delta;
      delete []bra_Dfull;
      
      delete []ket_Green;
      delete []ket_Delta;
      delete []ket_Dfull;
        
      delete []thread_integrand;
            
    }
    
    }
    
    /*
#pragma omp parallel for 
  for(int k=0; k<M; k++ ){ 
    //   r_type ek  = E_points[k];
    //integrand[k] *= 2.0/pow((1.0 - ek  * ek ),2.0);     //2.0/(IM_root[k]*IM_root[k]);//
    final_integrand[k] += integrand[k] ;
  }
    */
    
  delete []pre_factors;    
  delete []factors;
  delete []IM_energies;  
  delete []IM_root;
  //delete []integrand;    
}
