#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver_filtered.hpp"


void Kubo_solver_filtered::Greenwood_FFTs__imVec_eta_adjusted(std::complex<r_type> **bras, std::complex<r_type> **kets, r_type E_points[], r_type r_data[]){
  const std::complex<double> im(0,1);
  
  int nump    = parameters_.num_p_,
    size = parameters_.SECTION_SIZE_,
    decRate = filter_.parameters().decRate_,
    M_dec = parameters_.M_/decRate;

  
  r_type IM_root [nump];
  r_type IM_energies[nump];

  for(int m=0;m<nump;m++){
    IM_energies[m] = E_points[m];
    IM_root[m]     =  sqrt(1.0-IM_energies[m]*IM_energies[m]);
  }

    #pragma omp parallel 
    {

    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    


    
    if (id == Nthrds-1) l_end = size;

    
    fftw_plan plan1, plan2;      
    fftw_complex  *bra_f,  *ket_f, *input;
    r_type thread_data [nump];
    
    input   = ( fftw_complex* ) fftw_malloc(sizeof( fftw_complex ) * nump );    
    bra_f   = ( fftw_complex* ) fftw_malloc(sizeof( fftw_complex ) * nump );
    ket_f   = ( fftw_complex* ) fftw_malloc(sizeof( fftw_complex ) * nump );

    
    
    for(int m=0;m<nump;m++){
      input[m][0] = 0;
      input[m][1] = 0;  
    }

    for(int m=0;m<nump;m++)
      thread_data[m]=0;


# pragma omp critical
    {

      plan1 = fftw_plan_dft_1d(nump, input, bra_f, FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
      plan2 = fftw_plan_dft_1d(nump, input, ket_f, FFTW_BACKWARD, FFTW_ESTIMATE); //bra_derivative
    }

    for(int l=l_start; l<l_end;l++){

      for(int m=0;m<M_dec;m++){
	input[m][0] =   bras[m][l].real();
        input[m][1] =   bras[m][l].imag();
      }
    
      fftw_execute(plan1);
      
    
      for(int m=0;m<M_dec;m++){
	input[m][0] = kets[m][l].real();
	input[m][1] = kets[m][l].imag();
      }
     
      fftw_execute(plan2);   
      /*
      for(int j=0; j<nump; j++){
	bra[j]   = //( ( bra_re[j][0]     + im * bra_re[j][1]     ) / IM_root[j]   ).real()  -  im * ( ( bra_im[j][0]     + im * bra_im[j][1]     ) / IM_root[j]   ).real();       
	ket[j]   = //( ( ket_re[j][0]     + im * ket_re[j][1]     ) / IM_root[j]   ).real()  +  im * ( ( ket_im[j][0]     + im * ket_im[j][1]     ) / IM_root[j]   ).real();
      }
      */

      for(int m=0; m<nump; m++ )
        thread_data[ m ]  +=  bra_f[m][0] * ket_f[m][0] ;

    }


    # pragma omp critical
    {
      for(int m=0;m<nump;m++)
	r_data[m] += 2.0 * decRate * decRate * thread_data[m] / ( IM_root[m] * IM_root[m] );
    
      fftw_free(input);
      fftw_destroy_plan(plan1);
      fftw_free(bra_f);
      fftw_destroy_plan(plan2);
      fftw_free(ket_f);
      
    }
    
    }
    /*     
  if(parameters_.eta_!=0)
    for(int m=0;m<M;m++)
      r_data[m] *= sin(acos(E_points[m]));
}    */

}

