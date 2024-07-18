#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>


#include "fftw_wrapper.hpp"
#include "Kubo_solver_filtered.hpp"
    

void Kubo_solver_filtered::Greenwood_FFTs_imag(std::complex<r_type>** bras_re, std::complex<r_type>** bras_im, std::complex<r_type>** kets_re, std::complex<r_type>** kets_im, r_type r_data[]){  

  int nump    = parameters_.num_p_,
      size    = parameters_.SECTION_SIZE_,
      M_dec   = filter_.M_dec();
  
  const std::complex<double> im(0,1);  

  
#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    
    if (id == Nthrds-1)
      l_end = size;

    
    
    r_type thread_data [nump];
    for(int m = 0; m < nump; m++)
      thread_data[ m ] = 0;

    
    
    out_of_place_dft
      bras_re_dft( nump, BACKWARD ),
      kets_re_dft( nump, BACKWARD ),
      bras_im_dft( nump, BACKWARD ),
      kets_im_dft( nump, BACKWARD );
                     

    
    # pragma omp critical
    {
      bras_re_dft.create();
      kets_re_dft.create();
      bras_im_dft.create();
      kets_im_dft.create();
    } 

    //SO the solution here is to store separetely the two parts of each vector: real and imaginary. 
        
    for(int l = l_start; l < l_end;l++){      
      for(int m = 0; m < M_dec; m++){
	bras_re_dft.input()[ m ] = bras_re[ m ][ l ];
	kets_re_dft.input()[ m ] = kets_re[ m ][ l ];
	bras_im_dft.input()[ m ] = bras_im[ m ][ l ];
	kets_im_dft.input()[ m ] = kets_im[ m ][ l ];         
      }
      
      bras_re_dft.execute();
      kets_re_dft.execute();
      
      bras_im_dft.execute();
      kets_im_dft.execute();
      
      for(int m = 0; m < nump; m++ )
        thread_data[ m ] += real(     ( real( bras_re_dft( m ) ) - im * real( bras_im_dft( m ) ) ) * ( real( kets_re_dft( m ) ) + im * real( kets_im_dft( m ) ) )       );

			
    }


    # pragma omp critical
    {
      for(int m=0;m<nump;m++)
	r_data[m] +=  thread_data[m];
    }
  }
}




