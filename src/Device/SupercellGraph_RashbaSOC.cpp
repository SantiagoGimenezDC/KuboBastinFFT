#include "SupercellGraph_RashbaSOC.hpp"

#include<fstream>

void SupercellGraph_RashbaSOC::print_hamiltonian(){
  int dim = this->parameters().DIM_;
  int DIM = this->parameters().DIM_;

  Eigen::MatrixXcd H_r(dim,dim), S(dim,dim);

  std::ofstream dataP;
  dataP.open("velopx.txt");
        
    for(int j=0;j<dim;j++){
      for(int i=0;i<dim;i++){

	Eigen::Matrix<std::complex<double>, -1, 1>  term_i=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM), tmp=term_i, term_j=term_i, null=Eigen::Matrix<std::complex<double>, -1, 1>::Zero(DIM);
        Eigen::Matrix<double, -1, 1>  null2=Eigen::Matrix<double, -1, 1>::Zero(DIM);
	term_i(i)=1;
	term_j(j)=1;


        this->update_cheb(tmp.data(),term_j.data(),null.data());
	//this->H_ket(tmp.data(),term_j.data());
        //vel_op_x(tmp.data(),term_j.data());
        //vel_op_y(tmp.data(),term_j.data());
	std::complex<double> termy = term_i.dot(tmp);

	 H_r(i,j)=termy;
      }
    }

    dataP<<H_r;//.real();

    std::cout<<(H_r-H_r.transpose()).norm()<<std::endl;
  dataP.close();

  }


void SupercellGraph_RashbaSOC::traceover(type* traced, type* full_vec, int s, int num_reps){
  int subDim = this->parameters().SUBDIM_,
      C   = this->parameters().C_,
      W   = this->parameters().W_,
      sec_size = subDim/num_reps,
      buffer_length = sec_size;
	
  if( s == num_reps-1 )
      buffer_length += subDim % num_reps;


#pragma omp parallel for 
      for(int i=0;i<buffer_length;i++)
        traced[i] = full_vec[s*sec_size + i+2*C*W];

  };



void SupercellGraph_RashbaSOC::rearrange_initial_vec(type r_vec[]){ //supe duper hacky
  int Dim = this->parameters().DIM_,
    subDim = this->parameters().SUBDIM_;

  int C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;

  type tmp[subDim];

#pragma omp parallel for
    for(int n=0;n<subDim;n++)
      tmp[n]=r_vec[n];

#pragma omp parallel for
    for(int n=0;n<Dim;n++)
      r_vec[n] = 0;
        

#pragma omp parallel for
    for(int n=0;n<2*Le*W;n++)
      r_vec[2*C*W + n ]=tmp[ n];

}



void SupercellGraph_RashbaSOC::H_ket (r_type a, r_type b, type* ket, type* p_ket){

  int fullLe = ( this->parameters().LE_+2*this->parameters().C_ ),
      W      = this->parameters().W_;

  int C = parameters().C_,
      LE = parameters().LE_;

  r_type t = t_standard_/a,
         b_a = b/a,
         m_str = m_str_ * t,
         rashba_str = rashba_str_ * t;
  
  std::complex<r_type> f_y  = 2.0 * rashba_str * cos(M_PI/6.0)/3.0,
                       f_x  = 2.0 * rashba_str / 3.0,
                       f_x2 = 2.0 * rashba_str * sin(M_PI/6.0)/3.0;


  r_type * sFilter = damp_op();
  r_type * disorder_potential = dis();
  

  std::complex<r_type> ImUnit(0,1);
  f_y *= ImUnit;

#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      ket[ 2 * n ]     = ( b_a - m_str ) * p_ket[ 2 * n ] ;
      ket[ 2 * n + 1 ] = ( b_a + m_str ) * p_ket[ 2 * n + 1] ;


      
      if( i!=0 ){
	ket[ 2 * n ]     += ( -t * p_ket[ 2 * n - 2 ] + (  f_x2 * ((i%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 1 ] ) * peierls(i,-1);
	ket[ 2 * n + 1 ] += ( -t * p_ket[ 2 * n - 1 ] + ( -f_x2 * ((i%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 2 ] ) * peierls(i,-1);
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     += ( -t * p_ket[ 2 * n + 2 ] + (  f_x2 * ((i%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 3 ]   ) * peierls(i,1);
	ket[ 2 * n + 1 ] += ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * ((i%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 2 ]   ) * peierls(i,1);
      }
      if(j != (fullLe-1) && !(j == 0 && i == W-1) && i%2 == 0 ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2*n+2*(W+1) ]           +  f_x * p_ket[ 2 * n + 2 * (W+1) + 1 ] ) ;
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n + 2 * (W+1) + 1]  -  f_x * p_ket[ 2 * n + 2 * (W+1) ]);
      }
      if(j != 0  && i%2 != 0 ){
	ket[ 2 * n ]     +=  ( - t * p_ket[ 2 * n - 2 * ( W + 1 ) ]     -  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ] );
	ket[ 2 * n + 1 ] +=  ( - t * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ] +  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) ]   );
      }
      
      ket[ 2 * n ]     *= sFilter[ n ];
      ket[ 2 * n + 1 ] *= sFilter[ n ];	       
    }
 }



   if( disorder_potential != NULL )
     for( int i = W * C; i < W * C + W * LE ; i ++ ){
       ket[ 2 * i ]     += sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i ] / a;
       ket[ 2 * i +1 ]  += sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i + 1 ] / a;
     }



 if(CYCLIC_BCs_){

   //if(W%2!=0)
   // std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;

  

     for(int j=0; j<fullLe; j++){

       int n_up = j * W + W - 1;
       int n_down = j * W;


       ket[ 2 * n_up ]    +=  sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * (((0)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n_down + 1] ) * peierls(W-1,1);
       ket[ 2 * n_up + 1] +=  sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down + 1] + (  f_x2 * (((0)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n_down ]  ) * peierls(W-1,1);
      
      
       ket[ 2 * n_down]      +=  sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up ] + ( -f_x2 * ( ((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up + 1 ]   ) * peierls(0,-1);
       ket[ 2 * n_down + 1 ] +=  sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up + 1 ] + ( f_x2 * (((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up ]   ) * peierls(0,-1);
      
     } 


     //if(fullLe%2!=0)
     //std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

   for(int i=0; i<W; i++){
     int n_front = i;
     int n_back = (fullLe-1) * W + i;

      
     ket[ 2 * n_front ]     +=  sFilter[ n_front ] * ((n_front)%2 ==0 ) * ( -t * p_ket[ 2 * (n_back+1) ]    + ( -f_x * (((n_front)%2)==0? -1.0:1.0) ) * p_ket[ 2 * (n_back+1)  + 1 ] );
     ket[ 2 * n_front + 1 ] +=  sFilter[ n_front ] * ((n_front)%2 ==0 ) *( -t * p_ket[ 2 * (n_back+1) + 1 ] + ( f_x * (((n_front)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_back+1)  ]  );


     ket[ 2 * n_back ]     +=  sFilter[ n_back ] * ((n_back)%2 != 0) * ( -t * p_ket[ 2 * (n_front-1) ]     + ( f_x * (((n_back)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * ( n_front - 1 ) + 1 ]   );
     ket[ 2 * n_back + 1 ] +=  sFilter[ n_back ] * ((n_back)%2 != 0) * ( -t * p_ket[ 2 * (n_front-1) + 1 ] + ( -f_x * (((n_back)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * ( n_front - 1 ) ]   );
    }
 }
 
};









void SupercellGraph_RashbaSOC::update_cheb ( type ket[], type p_ket[], type pp_ket[]){

  int fullLe = ( this->parameters().LE_+2*this->parameters().C_ ),
      W      = this->parameters().W_;

  int C = parameters().C_,
      LE = parameters().LE_;

  r_type a = this->a(),
         b = this->b();

  r_type t = t_standard_/a,
         b_a = b/a,
         m_str = m_str_ * t,
         rashba_str = rashba_str_ * t;
  
  std::complex<r_type> f_y  = 2.0 * rashba_str * cos(M_PI/6.0)/3.0,
                       f_x  = 2.0 * rashba_str / 3.0,
                       f_x2 = 2.0 * rashba_str * sin(M_PI/6.0)/3.0;


  r_type * sFilter = damp_op();
  r_type * disorder_potential = dis();
  


  std::complex<r_type> ImUnit(0,1);
  
  f_y *= ImUnit;

  
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      ket[ 2 * n ]     =  ( ( b_a - m_str ) * p_ket[ 2 * n ] )    - 0.5 * sFilter[ n ] * pp_ket[ 2 * n ];
      ket[ 2 * n + 1 ] =  ( ( b_a + m_str ) * p_ket[ 2 * n + 1] ) - 0.5 * sFilter[ n ] * pp_ket[ 2 * n + 1 ];

      
      if( i!=0 ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2 * n - 2 ] + (  f_x2 * (((i)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 1 ] ) * peierls(i,-1);
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n - 1 ] + ( -f_x2 * (((i)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 2 ] ) * peierls(i,-1);
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2 * n + 2 ] + (  f_x2 * (((i)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 3 ] ) * peierls(i,1);
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * (((i)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 2 ] ) * peierls(i,1);
      }
      if(j != (fullLe-1) && !(j == 0 && i == W-1) && i%2 == 0 ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2*n+2*(W+1) ]           +  f_x * p_ket[ 2 * n + 2 * (W+1) + 1 ] ) ;
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n + 2 * (W+1) + 1]  -  f_x * p_ket[ 2 * n + 2 * (W+1) ]);
      }
      if(j != 0  && i%2 != 0 ){
	ket[ 2 * n ]     +=  ( - t * p_ket[ 2 * n - 2 * ( W + 1 ) ]     -  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ] );
	ket[ 2 * n + 1 ] +=  ( - t * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ] +  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) ]   );
      }
      
      ket[ 2 * n ]     *= 2.0 * sFilter[ n ];
      ket[ 2 * n + 1 ] *= 2.0 * sFilter[ n ];
      
    }
 }
 
   if( disorder_potential != NULL ){
     for( int i = W * C; i < W * C + W * LE ; i ++ ){
       ket[ 2 * i ]      +=  2 * sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i ] / a;
       ket[ 2 * i + 1 ]  +=  2 * sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i +1 ] / a;
     }    
   }
 


 
 if(CYCLIC_BCs_){
   //if(W%2!=0)
   // std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;


     for(int j=0; j<fullLe; j++){

       int n_up = j * W + W - 1;
       int n_down = j * W;


       ket[ 2 * n_up ]    += 2.0 * sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * (((0)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n_down + 1] ) * peierls(W-1,1);
       ket[ 2 * n_up + 1] += 2.0 * sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down + 1] + (  f_x2 * (((0)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n_down ]  ) * peierls(W-1,1);
      
      
       ket[ 2 * n_down]      += 2.0 * sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up ] + ( -f_x2 * ( ((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up + 1 ]   ) * peierls(0,-1);
       ket[ 2 * n_down + 1 ] += 2.0 * sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up + 1 ] + ( f_x2 * (((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up ]   ) * peierls(0,-1);
      
     } 


     //if(fullLe%2!=0)
     //std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 
   for(int i=0; i<W; i++){
     int n_front = i;
     int n_back = (fullLe-1) * W + i;

      
     ket[ 2 * n_front ]     += 2.0 * sFilter[ n_front ] * ((n_front)%2 == 0) * ( -t * p_ket[ 2 * (n_back+1) ]    + ( -f_x * (((n_front)%2)==0? -1.0:1.0) ) * p_ket[ 2 * (n_back+1) + 1 ] );
     ket[ 2 * n_front + 1 ] += 2.0 * sFilter[ n_front ] * ((n_front)%2 == 0) *( -t * p_ket[ 2 * (n_back+1) + 1 ] + ( f_x * (((n_front)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_back+1) ]  );


     ket[ 2 * n_back ]     += 2.0 * sFilter[ n_back ] * ((n_back)%2 != 0) * ( -t * p_ket[ 2 * (n_front-1) ]     + ( f_x * (((n_back)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_front-1) + 1 ]   );
     ket[ 2 * n_back + 1 ] += 2.0 * sFilter[ n_back ] * ((n_back)%2 != 0) * ( -t * p_ket[ 2 * (n_front-1) + 1 ] + ( -f_x * (((n_back)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_front-1) ]   );
    }

 }

#pragma omp parallel for 
 for(int n=0;n<2*fullLe*W;n++){
   pp_ket[n] = p_ket[n];
   p_ket[n]  = ket[n];
 }
 
}; 




void SupercellGraph_RashbaSOC::vel_op_x ( type* ket, type* p_ket){

  int Le = this->parameters().LE_,
      W      = this->parameters().W_,
    C      = this->parameters().C_,
    fullLe = Le+2*C;


  
  r_type t = this->a()*t_standard_  ;
 
  //if(this->parameters().C_%2==1)
  // std::cout<<"BEWARE VX OP ONLY WORKS FOR EVEN C!!!!!!!!!!"<<std::endl;


      
  std::complex<r_type>
    rashba_str = rashba_str_ * t,
    d_x  = a0_,
    d_x2 = a0_ * sin( M_PI /6.0 ),
    f_y  = 2.0 * rashba_str * cos( M_PI / 6.0 ) / 3.0,
    f_x  = 2.0 * rashba_str / 3.0,
    f_x2 = 2.0 * rashba_str * sin( M_PI / 6.0 ) / 3.0;


  std::complex<r_type> ImUnit(0,1);

  f_y *= ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;


        ket[ 2 * n ]     = 0;
	ket[ 2 * n + 1 ] = 0;

	

      if( n >= C * W ){
        if( i!=0 ){
	  ket[ 2 * n ]     += d_x2 * ((i%2)==0? -1.0:1.0) * ( -t * p_ket[ 2 * n - 2 ] + ( f_x2 * ((i%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 1 ] ) * peierls(i,-1);
	  ket[ 2 * n + 1 ] += d_x2 * ((i%2)==0? -1.0:1.0) * ( -t * p_ket[ 2 * n - 1] + ( -f_x2 * ((i%2)==0? -1.0:1.0)  + f_y ) * p_ket[ 2 * n - 2 ] ) * peierls(i,-1);
        }
        if(i != (W-1) ){
	  ket[ 2 * n ]     += d_x2 * ((i%2)==0? -1.0:1.0) * ( -t * p_ket[ 2 * n + 2 ] + ( f_x2 * ((i%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 3 ]   ) * peierls(i,1);
	  ket[ 2 * n + 1 ] += d_x2 * ((i%2)==0? -1.0:1.0) * ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * ((i%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 2 ]   ) * peierls(i,1);
        }
        if(j != (Le-1)  && !(j == C && i == W-1) && i%2==0 ){
	  ket[ 2 * n ]     +=   d_x * ( -t * p_ket[ 2 * n + 2 * ( W + 1 ) ]    -  f_x * p_ket[ 2 * n + 2 * ( W + 1 ) + 1 ]) ;
	  ket[ 2 * n + 1 ] +=   d_x * ( -t * p_ket[ 2 * n + 2 * ( W + 1 ) + 1 ]  +  f_x * p_ket[ 2 * n + 2 * ( W + 1 ) ]);
        }
        if(j != 0 && i%2!=0 ){
	  ket[ 2 * n ]     += -d_x * (  -t * p_ket[ 2 * n - 2 * ( W + 1 ) ]   +  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ]);
	  ket[ 2 * n + 1 ] += -d_x * (  -t * p_ket[ 2 * n - 2 * ( W + 1 ) + 1 ] -  f_x * p_ket[ 2 * n - 2 * ( W + 1 ) ] );
        }
      }
    }
 } 


 if(CYCLIC_BCs_ && C==0){
   //if( (W/2)%2 !=0 )
   //std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;
 
 
   for(int j=0; j<Le; j++){

     int n_up = j * W + W-1;
     int n_down = j * W;


     ket[ 2 * n_up ]     +=   d_x2 * (((W-1)%2)==0? -1.0:1.0) * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * (((W-1)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n_down + 1 ] ) * peierls(W-1,1);
     ket[ 2 * n_up + 1 ] +=   d_x2 * (((W-1)%2)==0? -1.0:1.0) *( -t * p_ket[ 2 * n_down + 1 ] + ( f_x2 * (((W-1)%2)==0? -1.0:1.0)  - f_y ) * p_ket[ 2 * n_down ]  ) * peierls(W-1,1);
      
      
     ket[ 2 * n_down ]     +=  -d_x2 * ((0%2)==0? -1.0:1.0) * ( -t * p_ket[ 2 * n_up ]     + ( -f_x2 * ((0%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up + 1 ]   ) * peierls(0,-1);
     ket[ 2 * n_down + 1 ] +=  -d_x2 * ((0%2)==0? -1.0:1.0) * ( -t * p_ket[ 2 * n_up + 1 ] + (  f_x2 * ((0%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up ]   ) * peierls(0,-1);
      
   } 


   //if(Le%2!=0)
   // std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 

   for(int i=0; i<W; i++){
     int n_front = i;
     int n_back = (Le-1) * W + i;

      
     ket[ 2 * n_front ]     +=  -d_x * std::complex<r_type>((n_front)%2 ==0) * ( -t * p_ket[ 2 * (n_back+1) ]     + ( f_x * (((n_front)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_back+1) + 1 ] );
     ket[ 2 * n_front + 1 ] +=  -d_x * std::complex<r_type>((n_front)%2 ==0) * ( -t * p_ket[ 2 * (n_back+1) + 1 ] + ( -f_x * (((n_front)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_back+1) ] );


     ket[ 2 * n_back ]     +=  d_x * std::complex<r_type>((n_back)%2 !=0) * ( -t * p_ket[ 2 * (n_front-1) ] + ( -f_x * (((n_back)%2)==0? -1.0:1.0) ) * p_ket[ 2 * (n_front-1) + 1 ]   );
     ket[ 2 * n_back + 1 ] +=  d_x * std::complex<r_type>((n_back)%2 !=0) * ( -t * p_ket[ 2 * (n_front-1) + 1 ] + ( f_x * (((n_back)%2)==0? -1.0:1.0)  ) * p_ket[ 2 * (n_front-1) ]   );
    }
 }



 
}; 



void SupercellGraph_RashbaSOC::vel_op_y ( type* ket, type* p_ket){

  int fullLe = ( this->parameters().LE_+2*this->parameters().C_ ),
      Le = this->parameters().LE_,
      W      = this->parameters().W_,
      C      = this->parameters().C_;


  
  r_type t = this->a()*t_standard_  ;
 
  //if(this->parameters().C_%2==1)
  // std::cout<<"BEWARE Vif(vertical_cbc) OP ONLY WORKS FOR EVEN C!!!!!!!!!!"<<std::endl;


      
  std::complex<r_type>
    rashba_str = rashba_str_ * t,
    d_y2 = a0_ * cos( M_PI / 6.0 ),
    f_y  = 2.0 * rashba_str * cos( M_PI / 6.0 ) / 3.0,
    f_x2 = 2.0 * rashba_str * sin( M_PI / 6.0 ) / 3.0;


  std::complex<r_type> ImUnit(0,1);

  f_y *= ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;
	
      ket[ 2 * n ]     = 0;
      ket[ 2 * n + 1 ] = 0;

      if( n >= C * W ){
        if( i!=0 ){
	  ket[ 2 * n ]     += - d_y2 * ( -t * p_ket[ 2 * n - 2 ] + (  f_x2 * (((i)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 1 ] ) * peierls(i,-1);
	  ket[ 2 * n + 1 ] += - d_y2 * ( -t * p_ket[ 2 * n - 1] +   ( -f_x2 * (((i)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n - 2 ] ) * peierls(i,-1);
        }
        if(i != ( W - 1 ) ){
	  ket[ 2 * n ]     += d_y2 * ( -t * p_ket[ 2 * n + 2 ] + (  f_x2 * (((i)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 3 ]   ) * peierls(i,1);
	  ket[ 2 * n + 1 ] += d_y2 * ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * (((i)%2)==0? -1.0:1.0) - f_y ) * p_ket[ 2 * n + 2 ]   ) * peierls(i,1);
        }
      }
    }
 } 




 if(CYCLIC_BCs_){
   //if( (W/2)%2 !=0 )
   // std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;
 
   for(int j=0; j<Le; j++){

     int n_up = j * W + W-1;
     int n_down = j * W;

     ket[ 2 * n_up ]     +=  -d_y2 *  ( -t * p_ket[ 2 * n_down ]     + ( -f_x2 * (((0)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_down + 1 ]   ) * peierls(W-1,1);
     ket[ 2 * n_up + 1 ] +=  -d_y2 *  ( -t * p_ket[ 2 * n_down + 1 ] + (  f_x2 * (((0)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_down ]   ) * peierls(W-1,1) ;
      
     ket[ 2 * n_down ]     +=  d_y2 *  ( -t * p_ket[ 2 * n_up ]     + ( -f_x2 * (((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up + 1 ]   ) * peierls(0,-1);
     ket[ 2 * n_down + 1 ] +=  d_y2 *  ( -t * p_ket[ 2 * n_up + 1 ] + (  f_x2 * (((W-1)%2)==0? -1.0:1.0) + f_y ) * p_ket[ 2 * n_up ]   ) * peierls(0,-1) ; 

   } 
 }
}; 


 




