#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver_SSD.hpp"



void fetch_buffer(
		  const std::string &filename,
		  const int M,
                  const int buffer,
		  const int size, 
		  type *matrix)
{

  for(int m=0;m<M;m++){
    FILE* in = fopen(filename.c_str(), "rb");
    fseek(in, ( buffer * size + m * size )   * sizeof(type), SEEK_CUR);
    fread( &matrix[m*size], 1, size*sizeof(type), in );
    fclose(in);
  }
  
}


void Kubo_solver_SSD::Greenwood_FFTs__imVec_noEta_SSD(std::complex<r_type> bras[], std::complex<r_type> kets[], SSD_buffer& bras_SSD, SSD_buffer& kets_SSD, r_type E_points[], r_type r_data[]){
  const std::complex<double> im(0,1);

  int M     = parameters_.M_,
      num_p = parameters_.num_p_;

 
  r_type kernel [ M ],
         pre_factor [ num_p ];


#pragma omp parallel for  
  for(int m=0;m<M;m++)
      kernel[m]      =  kernel_->term(m,M);
  

#pragma omp parallel for  
  for(int k=0;k<num_p;k++)
    pre_factor[k]    =  1.0-E_points[k]*E_points[k];


  
    
  int  total_read=0;
  int size = 0;
  
  for(int buffer_num = 0; buffer_num <= bras_SSD.num_buffers(); buffer_num++ ){
  
 
  auto read_start = std::chrono::steady_clock::now();
    
  bras_SSD.retrieve_row_buffer_from_SSD(buffer_num, bras);
  size = kets_SSD.retrieve_row_buffer_from_SSD(buffer_num, kets);
  
  auto read_end = std::chrono::steady_clock::now();  


  total_read += std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();

    
  if( size == 0 ) break;

    
      
  int Nthrds_bak  = omp_get_num_threads();
  if( size < Nthrds_bak )
    omp_set_num_threads(1);
  
  #pragma omp parallel 
    {
      int id, l_start, Nthrds, l_end;
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
	    input[m] = kernel[m] * bras[m*size+l].real();
     
	  fftw_execute(plan1);


	  for(int m=0;m<M;m++)
	    input[m] = kernel[m] * bras[m*size+l].imag();

	  fftw_execute(plan2);




      
	  for(int m=0;m<M;m++)
	    input[m] = kernel[m] * kets[m*size+l].real();

	
	  fftw_execute(plan3);
      
	  for(int m=0;m<M;m++)
	    input[m] = kernel[m] * kets[m*size+l].imag(); 

	  fftw_execute(plan4);   




	  for(int k=0; k<num_p; k++ )
	    thread_data[k] += (  (bra_re[k] - im * bra_im[k] )   *   ( ket_re[k] + im *  ket_im[k] )  ).real();
    
        }
      
    
      # pragma omp critical
      {
        for(int k=0;k<num_p;k++)
	  r_data[k] += 2.0 * thread_data[k] / pre_factor[k] ;

      
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
    
    if( size < Nthrds_bak )
      omp_set_num_threads(Nthrds_bak);
  
  }

  int SEC_SIZE     = parameters_.SECTION_SIZE_;
  
  std::cout<<"              Time spent reading SSD buffer:    "<<total_read/1000<<"ms "<<std::endl;
  std::cout<<"              Average SSD download bandwidth:   "<<   double( 2 * double(SEC_SIZE) * double(M) * sizeof(type) )/ (double(total_read) *1000)<<" GB/s" <<std::endl;
}

