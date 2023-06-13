#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver.hpp"


void Kubo_solver::Bastin_FFTs__imVec_noEta_2(std::complex<r_type> **bras, std::complex<r_type> **kets, r_type E_points[], r_type final_integrand[]){

  const std::complex<double> im(0,1);  

  int M  = parameters_.M_,
    size = parameters_.SECTION_SIZE_;

    

  std::complex<r_type> *factors = new std::complex<r_type> [M];
  r_type *IM_root = new r_type [M],
   *integrand     = new r_type [M];

  for(int m=0;m<M;m++)
    integrand[m]=0;


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

    if (id == Nthrds-1) l_end = size;

    


    //8 planos+ 14 vetores [M] por thread. Pode ser reduzido a 2 planos e uns 4 vetores?
    
    fftw_plan plan1, plan2, plan3, plan4,
              plan5, plan6, plan7, plan8;

    std::complex<r_type>
      bra_Green[M],
      ket_Green[M];
    
    /*
    std::complex<r_type>
      *bra_Green = new std::complex<r_type> [M],
      *bra_Delta = new std::complex<r_type> [M],
      *bra_Dfull = new std::complex<r_type> [M],
      
      *ket_Green = new std::complex<r_type> [M],
      *ket_Delta = new std::complex<r_type> [M],
      *ket_Dfull = new std::complex<r_type> [M];   
      */
    
    fftw_complex   
      *bra_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *bra_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),

      *bra_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *bra_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),


      *ket_re   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *ket_im   = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),

      *ket_D_re = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M ),
      *ket_D_im = ( fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * M );


    std::complex<r_type>         
      p[M], w[M];

    for(int k=0;k<M;k++){
      p[k] = 0;
      w[k] = 0;
    }


      

    
# pragma omp critical
    {
      plan1 = fftw_plan_dft_1d(M, bra_re,   bra_re,    FFTW_BACKWARD, FFTW_ESTIMATE); 
      plan2 = fftw_plan_dft_1d(M, bra_im,   bra_im,    FFTW_BACKWARD, FFTW_ESTIMATE); 
      plan3 = fftw_plan_dft_1d(M, bra_D_re, bra_D_re,  FFTW_BACKWARD, FFTW_ESTIMATE); 
      plan4 = fftw_plan_dft_1d(M, bra_D_im, bra_D_im,  FFTW_BACKWARD, FFTW_ESTIMATE); 

      plan5 = fftw_plan_dft_1d(M, ket_re,   ket_re,    FFTW_BACKWARD, FFTW_ESTIMATE); 
      plan6 = fftw_plan_dft_1d(M, ket_im,   ket_im,    FFTW_BACKWARD, FFTW_ESTIMATE); 
      plan7 = fftw_plan_dft_1d(M, ket_D_re, ket_D_re,  FFTW_BACKWARD, FFTW_ESTIMATE); 
      plan8 = fftw_plan_dft_1d(M, ket_D_im, ket_D_im,  FFTW_BACKWARD, FFTW_ESTIMATE); 

    }

    for(int l=l_start; l<l_end;l++){
      for(int m=0;m<M;m++){

	bra_Green[m] = bras[m][l]; //This single access row-wise access of a col-major matrix is the longest operation in this loop!!
	
	bra_re[m][0] = ( factors[m] ).real() * bra_Green[m].real(); 
	bra_re[m][1] = ( factors[m] ).imag() * bra_Green[m].real(); 


	bra_im[m][0] = ( factors[m] ).real() * bra_Green[m].imag(); 
	bra_im[m][1] = ( factors[m] ).imag() * bra_Green[m].imag(); 


	bra_D_re[m][0] = m * bra_re[m][0]; // m * ( conj ( factors[m] * bra_Green[m] ) ).real(); // 
	bra_D_re[m][1] = m * bra_re[m][1]; //m * ( conj ( factors[m] * bra_Green[m] ) ).imag(); // 


	bra_D_im[m][0] = m * bra_im[m][0]; 
	bra_D_im[m][1] = m * bra_im[m][1]; 




	ket_Green[m] = kets[m][l]; 	

	ket_re[m][0] = ( factors[m] ).real() * ket_Green[m].real(); 
	ket_re[m][1] = ( factors[m] ).imag() * ket_Green[m].real(); 


	ket_im[m][0] = ( factors[m] ).real() * ket_Green[m].imag(); 
	ket_im[m][1] = ( factors[m] ).imag() * ket_Green[m].imag(); 


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
        p[2*j] += ( bra_re[j][0] - im * bra_im[j][0] )     *       ( ket_re  [j][0] + im * ket_re  [j][1] + im * ( ket_im  [j][0] + im * ket_im  [j][1] ) ) +
	          ( bra_re[j][0] - im * bra_re  [j][1] - im * ( bra_im  [j][0] - im * bra_im  [j][1] ) )    *    ( ket_re[j][0] + im * ket_im[j][0] );


	w[2*j] += ( bra_D_re[j][0] - im * bra_D_re[j][1] - im * ( bra_D_im[j][0] - im * bra_D_im[j][1] ) )    *    ( ket_re[j][0] + im * ket_im[j][0] ) -
	          ( bra_re[j][0] - im * bra_im[j][0] )     *     ( ket_D_re[j][0] + im * ket_D_re[j][1] + im * ( ket_D_im[j][0] + im * ket_D_im[j][1] ) );



	
	p[2*j+1] += ( bra_re[M-j-1][0] - im * bra_im[M-j-1][0] )    *    ( ket_re[M-j-1][0] - im * ket_re  [M-j-1][1] + im * ( ket_im  [M-j-1][0] - im * ket_im  [M-j-1][1] ) ) +
	            ( bra_re[M-j-1][0] + im * bra_re  [M-j-1][1] - im * ( bra_im  [M-j-1][0] + im * bra_im  [M-j-1][1] ) )    *    ( ket_re[M-j-1][0] + im * ket_im[M-j-1][0] );


	w[2*j+1] += ( bra_D_re[M-j-1][0] + im * bra_D_re[M-j-1][1] - im * ( bra_D_im[M-j-1][0] + im * bra_D_im[M-j-1][1] ) )    *    ( ket_re[M-j-1][0] + im * ket_im[M-j-1][0] ) -
	            ( bra_re[M-j-1][0] - im * bra_im[M-j-1][0] )    *    ( ket_D_re[M-j-1][0] - im * ket_D_re[M-j-1][1] + im * ( ket_D_im[M-j-1][0] - im * ket_D_im[M-j-1][1] ) );
	  
      }
      /*
      for(int j=0; j<M/2; j++){
	
	bra_Delta[2*j] = bra_re  [j][0]                       - im *   bra_im  [j][0];	
	bra_Green[2*j] = bra_re  [j][0] - im * bra_re  [j][1] - im * ( bra_im  [j][0] - im * bra_im  [j][1] );//BRA is conjugated for the dot product
	bra_Dfull[2*j] = bra_D_re[j][0] - im * bra_D_re[j][1] - im * ( bra_D_im[j][0] - im * bra_D_im[j][1] );

	ket_Delta[2*j] = ket_re  [j][0]                       + im *   ket_im  [j][0];	
	ket_Green[2*j] = ket_re  [j][0] + im * ket_re  [j][1] + im * ( ket_im  [j][0] + im * ket_im  [j][1] );
	ket_Dfull[2*j] = ket_D_re[j][0] + im * ket_D_re[j][1] + im * ( ket_D_im[j][0] + im * ket_D_im[j][1] );


	
	bra_Delta[2*j+1] = bra_re  [M-j-1][0]                           - im *   bra_im  [M-j-1][0];	
	bra_Green[2*j+1] = bra_re  [M-j-1][0] + im * bra_re  [M-j-1][1] - im * ( bra_im  [M-j-1][0] + im * bra_im  [M-j-1][1] );//FFT algo conversion forces conjugation in re and im parts
	bra_Dfull[2*j+1] = bra_D_re[M-j-1][0] + im * bra_D_re[M-j-1][1] - im * ( bra_D_im[M-j-1][0] + im * bra_D_im[M-j-1][1] );

	ket_Delta[2*j+1] = ket_re  [M-j-1][0]                           + im * ( ket_im  [M-j-1][0]  );	
	ket_Green[2*j+1] = ket_re  [M-j-1][0] - im * ket_re  [M-j-1][1] + im * ( ket_im  [M-j-1][0] - im * ket_im  [M-j-1][1] );//FFT algo conversion forces conjugation in re and im parts
	ket_Dfull[2*j+1] = ket_D_re[M-j-1][0] - im * ket_D_re[M-j-1][1] + im * ( ket_D_im[M-j-1][0] - im * ket_D_im[M-j-1][1] );
      }
      


      for(int k=0; k<M; k++ ){
        p[k] += bra_Delta[k] * ket_Green[k] + bra_Green[k] * ket_Delta[k];
        w[k] += bra_Dfull[k] * ket_Delta[k] - bra_Delta[k] * ket_Dfull[k];
	} */         
    }

    
    # pragma omp critical
    {
      for(int k=0;k<M;k++)
	final_integrand[k] += E_points[k] * (p[k]).real() + ( im * IM_root[k] * w[k] ).real();
      
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
      /*
      delete []bra_Green;
      delete []bra_Delta;
      delete []bra_Dfull;
      
      delete []ket_Green;
      delete []ket_Delta;
      delete []ket_Dfull;
      */

      
      
    }
    
    }

        
#pragma omp parallel for 
  for(int k=0; k<M; k++ ){
    r_type ek  = E_points[k];
    /*    if(sqrt(ek*ek)>0.99 || sqrt(ek*ek)<0.01){
     final_integrand[k] = 0;
    }
    else*/
      final_integrand[k] *= 2.0/pow((1.0 - ek  * ek ),2.0);
  }


  delete []factors;
  delete []IM_root;
  delete []integrand;    
}
