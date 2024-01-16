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


void Kubo_solver_SSD::Greenwood_FFTs__imVec_noEta_SSD(std::complex<r_type> bras[], std::complex<r_type> kets[], r_type E_points[], r_type r_data[]){
  const std::complex<double> im(0,1);

  int SEC_SIZE = parameters_.SECTION_SIZE_,
      M     = parameters_.M_,
      size  = SEC_SIZE/num_buffers_,
      num_p = parameters_.num_p_;

  int stride=size;
 
  r_type kernel [ M ],
         pre_factor [ num_p ];


#pragma omp parallel for  
  for(int m=0;m<M;m++)
      kernel[m]      =  kernel_->term(m,M);
  

#pragma omp parallel for  
  for(int k=0;k<num_p;k++)
    pre_factor[k]    =  1.0-E_points[k]*E_points[k];


  
  
  type *bras_l = new type [ size * M ],
       *kets_l = new type [ size * M ];
  
  int  total_read=0;
  
for(int buffer = 0; buffer < num_buffers_; buffer++ ){

  std::string filename=parameters_.run_dir_+"/buffer/";
  
  
  if(buffer == (num_buffers_ -1) ){
    if (SEC_SIZE % num_buffers_ != 0) {
     size = SEC_SIZE % num_buffers_;
     //   std::cout<<"yes"<<std::endl;
    }
    else if(SEC_SIZE % num_buffers_ == 0)
     break;
  }

  
  auto read_start = std::chrono::steady_clock::now();

  for(int m=0;m<M;m++){
    FILE* in = fopen((filename+"brass").c_str(), "rb");
    fseek(in, ( buffer * stride + m * SEC_SIZE )   * sizeof(type), SEEK_CUR);
    fread( &bras_l[m*stride], 1, size*sizeof(type), in );
    fclose(in);
  }

  for(int m=0;m<M;m++){
    FILE* in = fopen((filename+"ketss").c_str(), "rb");
    fseek(in, ( buffer * stride + m * SEC_SIZE )   * sizeof(type), SEEK_CUR);
    fread( &kets_l[m*stride], 1, size*sizeof(type), in );
    fclose(in);
  }


  //fetch_buffer( filename+"brass", M, buffer, size, bras_l);
  //fetch_buffer( filename+"ketss", M, buffer, size, kets_l);
  auto read_end = std::chrono::steady_clock::now();  

  total_read += std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();

  
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
	    input[m] = kernel[m] * bras_l[m*size+l].real();
     
	  fftw_execute(plan1);


	  for(int m=0;m<M;m++)
	    input[m] = kernel[m] * bras_l[m*size+l].imag();

	  fftw_execute(plan2);




      
	  for(int m=0;m<M;m++)
	    input[m] = kernel[m] * kets_l[m*size+l].real();

	
	  fftw_execute(plan3);
      
	  for(int m=0;m<M;m++)
	    input[m] = kernel[m] * kets_l[m*size+l].imag(); 

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

 std::cout<<"           Time spent reading SSD buffer:   "<<total_read/1000<<"ms "<<std::endl;

 delete [] bras_l;
 delete [] kets_l;
}

