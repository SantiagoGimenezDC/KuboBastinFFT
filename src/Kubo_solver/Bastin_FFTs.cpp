#include<iostream>
#include<cmath>
#include<omp.h>
#include<thread>
#include<complex>

#include<fftw3.h>

#include "Kubo_solver_filtered.hpp"
#include "fftw_wrapper.hpp"

  
  /*================================================================================================================================================//
  //   FFTW variables; The real and imaginary parts of the random vector need to be processed separately, as if they were different random vectors; //
  //   The reason for that is, FFTW provides the Fourier transform as e^(2*i*pi*m*k). We need the transform proportional to e^(i*pi*m*k),           //
  //   without the factor 2 within the exponent. Hence, a change of variable k=2j is needed (Weisse/2006), and such change of variables             //
  //   requires conjugating the  FFT output for odd entries. Conjugation would act over the imaginary part of the rand vec too if done naively,     //
  //   giving wrong results. Hence, to preserve the imaginary part of the randVec, it is necessary to treat it separetely. At  least thats the      //
  //   only way I found of doing it.                                                                                                                //
  //================================================================================================================================================*/

inline type derivate( r_type E_points[], out_of_place_dft& series, int j, int nump, std::vector<int> sign){
  type delta = 1.0, der=0;

  if( j == nump / 2 )
    der=0;
  else{
      type j_v = series(j),
	   jp1_v = series(j+1);
      

      if( sign[j] == -1) j_v = conj(j_v);
      if( sign[j+1] == -1 ) jp1_v = conj(jp1_v);

      
    if( j < nump - 1 ){
      delta = E_points[j+1]-E_points[j];
      der = ( jp1_v - j_v ) / delta;
    }
    if( j == nump - 1 ){
      delta = E_points[0]-E_points[j];            
      der = ( jp1_v - j_v ) / delta;
    }
  }
  /*
  if( j < nump/2 )
    der = ( series(j+1)-series(j) ) / delta;
  else if ( ( j != nump - 1 ) && ( j > nump/2 ) )
    der = ( series(j)-series(j+1) ) / delta;
  else // if( j == nump-1)
    der = 0;
      */
  return der; 

} 


inline type derivate_2nd_order( r_type E_points[], out_of_place_dft& series, int j, int nump, std::vector<int> sign){
  type der=0, x0=0, x1=0, x2=0, term1=0, term2=0, term3=0;

  if( j == nump / 2 || j == nump/2 + 1 )
    der=0;
  else{    
    /*if( j == 0 ){
      type j_v = series(j),
	   jp1_v = series(j+1),
	   jm1_v = series(nump-1);

      if( sign[j] == -1 ) j_v = conj(j_v);
      if( sign[j+1] == -1 ) jp1_v = conj(jp1_v);
      if( sign[nump-1] == -1 ) jm1_v = conj(jm1_v);


      x0 = E_points[nump-1];
      x1 = E_points[j];
      x2 = E_points[j+1];
        
      term1 = jm1_v * ( (x1 - x2) / ((x0 - x1) * (x0 - x2)) );
      term2 = j_v * (2 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2));
      term3 = jp1_v * (x1 - x0) / ((x2 - x0) * (x2 - x1));
      
      der = term1 + term2 + term3;
    } 
    else*/
      if( j < nump - 1 ){
      type jm1_v = series(j-1),
	   j_v = series(j),
           jp1_v = series(j+1);

      if( sign[j-1] == -1 ) jm1_v = conj(jm1_v);      
      if( sign[j] == -1 ) j_v = conj(j_v);
      if( sign[j+1] == -1 ) jp1_v = conj(jp1_v);



      x0 = E_points[j-1];
      x1 = E_points[j];
      x2 = E_points[j+1];
        
      term1 = jm1_v * ( (x1 - x2) / ((x0 - x1) * (x0 - x2)) );
      term2 = j_v * (2 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2));
      term3 = jp1_v * (x1 - x0) / ((x2 - x0) * (x2 - x1));
      
      der = term1 + term2 + term3;
    }/*
    else if( j == nump - 1 ){
      type j_v = series(j),
	   jp1_v = series(0),
	   jm1_v = series(j-1);

      if( sign[j] == -1 ) j_v = conj(j_v);
      if( sign[0] == -1 ) jp1_v = conj(jp1_v);
      if( sign[j-1] == -1 ) jm1_v = conj(jm1_v);


      x0 = E_points[j-1];
      x1 = E_points[j];
      x2 = E_points[0];
        
      term1 = jm1_v * ( (x1 - x2) / ((x0 - x1) * (x0 - x2)) );
      term2 = j_v * (2 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2));
      term3 = jp1_v * (x1 - x0) / ((x2 - x0) * (x2 - x1));
      
      der = term1 + term2 + term3;
      }*/
  }

  return der; 

} 

void Kubo_solver_filtered::Bastin_FFTs  (r_type E_points[], std::complex<r_type>** bras, std::complex<r_type>** kets,  type* r_data, int s){

  const std::complex<double> im(0,1);
  
  int M    = parameters_.M_,
      size = parameters_.SECTION_SIZE_,
       nump = parameters_.num_p_,
       num_parts = parameters_.num_parts_;
  
  int M_dec = filter_.M_dec(),
      M_ext = filter_.parameters().M_ext_,
      L = filter_.parameters().L_,
      Np = (L-1)/2;

  std::vector<int> list = filter_.decimated_list(),
    sign(nump);  


  for(int i =0; i<nump;i++){
    double E_prev ;

    if( i == 0 && nump == M_ext )
       E_prev = E_points[ nump - 1 ];
    else
         E_prev = E_points[ i - 1 ];
    
    if( E_points[i] - E_prev < 0 )
      sign[i] = 1;
    else
      sign[i] = -1;
  }

  
  if( s != num_parts-1)
    size -= device_.parameters().SUBDIM_ % num_parts;

      


#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    if (id == Nthrds-1)
      l_end = size;
  
  
  //8 plans + 14 [M] sized vectors per thread. Can it be reduced to 2 plans and some 4 vectors?

    out_of_place_dft
      bras_dft( nump, BACKWARD ),
      kets_dft( nump, BACKWARD );


   std::complex<double> D_bras, D_kets, //variables to access values within bras/kets         
                        p[nump], w[nump]; //Dot product partial results;

   for(int k = 0; k < nump; k++){
     p[k] = 0;
     w[k] = 0;
   }
    

    //FFTW plans. All plans are in-place backward 1D FFTS of the entry variables; These functions are no thread safe for some reason, hence the # pragma critical
# pragma omp critical
   {
     bras_dft.create();
     kets_dft.create();
   }

   for(int l = l_start; l < l_end; l++){
      if( M_ext > M + Np ){
        int m = 0;
        while( list[m] < M + Np){
	  bras_dft.input()[ m ] = bras[ m ][ l ];
	  kets_dft.input()[ m ] = kets[ m ][ l ];
          m++;
        }

        m = 0;
        while( list[M_dec - 1 - m ] > M_ext - 1 - Np ){
	  bras_dft.input()[ nump - 1 - m ] = bras[ M_dec - 1 - m ][ l ];
	  kets_dft.input()[ nump - 1 - m ] = kets[ M_dec - 1 - m ][ l ];
          m++;
        }
      }
      else	
        for(int m = 0; m < nump; m++){
	  bras_dft.input()[ m ] = bras[ m ][ l ];
	  kets_dft.input()[ m ] = kets[ m ][ l ];
	}

     bras_dft.execute();
     kets_dft.execute();



     for(int j = 0; j < nump; j++){

       D_bras = derivate_2nd_order( E_points, bras_dft, j, nump, sign);
       D_kets = derivate_2nd_order( E_points, kets_dft, j, nump, sign);

       if( sign[j] == 1 ){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += //real( bras_dft( j ) ) * real( kets_dft( j ) );

	 real( bras_dft(j) ) * ( kets_dft(j)   ) + //Re(G(k)) * G(k)+
	 conj( bras_dft(j) ) * real( kets_dft(j) );   //G(k) * Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
       w[j] +=   D_bras   *  real( kets_dft(j) ) - //dG(k) * Re(G(k))-	  
                real( bras_dft(j) )  *  D_kets ; //Re(G(k)) * dG(k)
       }

       if( sign[j] == -1 ){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( bras_dft(j) )  ) * ( conj(kets_dft(j) )   ) + // //Re(G(k)) *G(k)+
	         ( bras_dft(j)     ) * ( real( kets_dft(j) )  );   ////G(k)Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += (  D_bras   )  * ( real( kets_dft(j)   )  ) - ////dG(k)Re(G(k))
		 ( real( bras_dft(j) )    )  * (  D_kets   ); // //Re(G(k))dG(k)

       }
     }
   }
     /*
     for(int j = 0; j < nump; j++){
       
       D_bras = derivate( E_points, bras_dft, j, nump);
       D_kets = derivate( E_points, kets_dft, j, nump);
       
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] +=  real( bras_dft(j) ) * ( kets_dft(j)   ) + //Re(G(k)) * G(k)+
	        conj( bras_dft(j) )  *  real( kets_dft(j) );   //G(k) * Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
       w[j] +=  conj( D_bras )  *  real( kets_dft(j) ) - //dG(k) * Re(G(k))-	  
		real( bras_dft(j) )  *  D_kets ; //Re(G(k)) * dG(k)
     }

     }   */   
    
# pragma omp critical
   {
     for(int k = 0; k < nump; k++){
       r_data[ k ]        += p[ k ];
       r_data[ k + nump ] += w[ k ];	
     }
   } 
 }
}
