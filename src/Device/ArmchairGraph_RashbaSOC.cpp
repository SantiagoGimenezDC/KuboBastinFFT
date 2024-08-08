#include "ArmchairGraph_RashbaSOC.hpp"


void ArmchairGraph_RashbaSOC::traceover(type* traced, type* full_vec, int s, int num_reps){
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



void ArmchairGraph_RashbaSOC::rearrange_initial_vec(type r_vec[]){ //supe duper hacky
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


void ArmchairGraph_RashbaSOC::RashbaSOC_Hamiltonian (SpMatrixXpc& H ){

  int fullLe = ( this->parameters().LE_+2*this->parameters().C_ ),
    W      = this->parameters().W_;

 int C = parameters().C_,
   LE = parameters().LE_;

  r_type t = t_standard_,
         rashba_str = rashba_str_ * t,
         m_str = t * m_str_;
    

    
  std::complex<r_type> f_y  = 2.0 * rashba_str * cos( M_PI /6.0 ) / 3.0,
                       f_x  = 2.0 * rashba_str / 3.0,
                       f_x2 = 2.0 * rashba_str*sin( M_PI / 6 ) / 3.0;

  //  Eigen::MatrixXi nonContacts = this->nonContactPoints();

  std::complex<r_type> ImUnit(0,1);


  //f_x*=ImUnit;
  //f_x2*=ImUnit;
  f_y*=ImUnit;


  r_type* disorder_potential = this->dis();
  

 
  typedef Eigen::Triplet<std::complex<r_type>> Tc;
  std::vector<Tc> tripletList;
  tripletList.reserve(5*fullLe*W);
  



  //#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      tripletList.push_back( Tc( 2*n, 2*n, -m_str ) );
      tripletList.push_back( Tc( (2*n+1), (2*n+1), m_str ) );	
      

      
      if( i!=0 ){
	tripletList.push_back(Tc((2*n),(2*n-2),-t));
	tripletList.push_back(Tc((2*n),(2*n-1),  ( +f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1)  + f_y ) ));

	tripletList.push_back(Tc((2*n+1),(2*n-1),-t));
	tripletList.push_back(Tc((2*n+1),(2*n-2),  ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1)  + f_y ) ));
	
      }
      if(i != (W-1) ){
	tripletList.push_back(Tc((2*n),(2*n+2),-t));
	tripletList.push_back(Tc((2*n),(2*n+3), ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? -1:1) - f_y )  ));

	tripletList.push_back(Tc((2*n+1),(2*n+3),-t));
	tripletList.push_back(Tc((2*n+1),(2*n+2),   ( f_x2 * std::complex<r_type>(((j+i)%2)==0? -1:1) - f_y )    ));

      }
      if(j != (fullLe-1)){
	tripletList.push_back(Tc((2*n),(2*n+2*W),  -(std::complex<r_type>((j+i)%2)) * t ) );
	tripletList.push_back(Tc((2*n),(2*n+2*W+1), (std::complex<r_type>((j+i)%2)) * f_x ) );

	tripletList.push_back(Tc((2*n+1),(2*n+2*W+1), -std::complex<r_type>((j+i)%2) * t ) );
	tripletList.push_back(Tc((2*n+1),(2*n+2*W),  -std::complex<r_type>((j+i)%2) * f_x  ) );

      }

      if(j != 0){
	tripletList.push_back(Tc((2*n),(2*n-2*W),     -std::complex<r_type>((j+i+1)%2) * t));
	tripletList.push_back(Tc((2*n),(2*n-2*W+1),   -std::complex<r_type>((j+i+1)%2) * f_x  ));

	tripletList.push_back(Tc((2*n+1),(2*n-2*W+1), -std::complex<r_type>((j+i+1)%2) * t));
	tripletList.push_back(Tc((2*n+1),(2*n-2*W),    std::complex<r_type>((j+i+1)%2) * f_x  ));
       }
    }
 }  

 if(disorder_potential != NULL){
   for(int i = W*C; i < W * C + W * LE ; i++){
     tripletList.push_back( Tc( i,i,disorder_potential[ i - W*C ] ) );
     tripletList.push_back( Tc( i+1,i+1,disorder_potential[ i - W*C ] ) );
   }
 }    
     
 

  
 H.resize(fullLe*W, fullLe*W);

 H.setFromTriplets(tripletList.begin(), tripletList.end(),[] (const std::complex<r_type> &a,const std::complex<r_type> &b) { return a+b; });


}; 



void ArmchairGraph_RashbaSOC::update_cheb ( type* ket, type* p_ket, type* pp_ket){

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
  //  f_x*=ImUnit;
  // f_x2*=ImUnit;

  f_y *= ImUnit;


#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      ket[ 2 * n ]     =  2.0 * ( ( b_a - m_str ) * p_ket[ 2 * n ] )    - sFilter[ n ] * pp_ket[ 2 * n ];
      ket[ 2 * n + 1 ] =  2.0 * ( ( b_a + m_str ) * p_ket[ 2 * n + 1] ) - sFilter[ n ] * pp_ket[ 2 * n + 1 ];

      
      if( i!=0 ){
	ket[ 2 * n ]     += 2.0 * ( -t * p_ket[ 2 * n - 2 ] + (  f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 1 ] );
	ket[ 2 * n + 1 ] += 2.0 * ( -t * p_ket[ 2 * n - 1 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 2 ] );
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     += 2.0 * ( -t * p_ket[ 2 * n + 2 ] + (  f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 3 ]   );
	ket[ 2 * n + 1 ] += 2.0 * ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 2 ]   );
      }
      if(j != (fullLe-1)){
	ket[ 2 * n ]     += 2.0 * ((j+i)%2) * ( -t * p_ket[ 2*n+2*W ]           +  f_x * p_ket[ 2 * n + 2 * W + 1 ] ) ;
	ket[ 2 * n + 1 ] += 2.0 * ((j+i)%2) * ( -t * p_ket[ 2 * n + 2 * W + 1]  -  f_x * p_ket[ 2 * n + 2 * W ]);
      }
      if(j != 0){
	ket[ 2 * n ]     += 2.0 * ((j+i+1)%2) * ( - t * p_ket[ 2 * n - 2 * W ]     -  f_x * p_ket[ 2 * n - 2 * W + 1 ] );
	ket[ 2 * n + 1 ] += 2.0 * ((j+i+1)%2) * ( - t * p_ket[ 2 * n - 2 * W + 1 ] +  f_x * p_ket[ 2 * n - 2 * W ]   );
      }
      
      ket[ 2 * n ]     *= sFilter[ n ];
      ket[ 2 * n + 1 ] *= sFilter[ n ];	       
    }
 }


 
   if( disorder_potential != NULL ){
     for( int i = W * C; i < W * C + W * LE ; i ++ ){
       ket[ 2 * i ]      +=  2 * sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i ] / a;
       ket[ 2 * i + 1 ]  +=  2 * sFilter[ i ] * disorder_potential[ i - W * C ] * p_ket[ 2 * i +1 ] / a;
     }    
   }
 

 
 bool vertical_cbc=false;
 bool horizontal_cbc=false;


  if(W%2!=0&&vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;

  
 if(vertical_cbc)
  for(int j=0; j<fullLe; j++){

    int n_up = j * W + W - 1;
    int n_down = j * W;


    ket[ 2 * n_up ]    += 2.0 * sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n_down + 1] );
    ket[ 2 * n_up + 1] += 2.0 * sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down + 1] + (  f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n_down ]  );
      
      
    ket[ 2 * n_down]      += 2.0 * sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up ] + ( -f_x2 * std::complex<r_type>( ((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up + 1 ]   );
    ket[ 2 * n_down + 1 ] += 2.0 * sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up + 1 ] + ( f_x2 * std::complex<r_type>(((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up ]   );
      
 } 


 if(fullLe%2!=0&&horizontal_cbc)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 
 if(horizontal_cbc)
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (fullLe-1) * W + i;

      
	ket[ 2 * n_front ]     += 2.0 * sFilter[ n_front ] * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_back ]    + ( -f_x * std::complex<r_type>(((n_front)%2)==0? 1:-1) ) * p_ket[ 2 * n_back + 1 ] );
	ket[ 2 * n_front + 1 ] += 2.0 * sFilter[ n_front ] * ((n_front+1)%2) *( -t * p_ket[ 2 * n_back + 1 ] + ( f_x * std::complex<r_type>(((n_front)%2)==0? 1:-1)  ) * p_ket[ 2 * n_back ]  );


	ket[ 2 * n_back ]     += 2.0 * sFilter[ n_back ] * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_front ]     + ( f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1)  ) * p_ket[ 2 * n_front + 1 ]   );
	ket[ 2 * n_back + 1 ] += 2.0 * sFilter[ n_back ] * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_front + 1 ] + ( -f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1)  ) * p_ket[ 2 * n_front ]   );

    }

#pragma omp parallel for 
 for(int n=0;n<2*fullLe*W;n++){
   pp_ket[n] = p_ket[n];
   p_ket[n]  = ket[n];
 }

 
}; 




void ArmchairGraph_RashbaSOC::vel_op ( type* ket, type* p_ket){

  int Le = this->parameters().LE_,
      W      = this->parameters().W_,
      C      = this->parameters().C_;


  
  r_type t = t_standard_  ;
 
  if(this->parameters().C_%2==1)
    std::cout<<"BEWARE VX OP ONLY WORKS FOR EVEN C!!!!!!!!!!"<<std::endl;


      
  std::complex<r_type>
    rashba_str = rashba_str_ * t,
    d_x  = a0_,
    d_x2 = a0_ * sin( M_PI /6.0 ),
    f_y  = 2.0 * rashba_str * cos( M_PI / 6.0 ) / 3.0,
    f_x  = 2.0 * rashba_str / 3.0,
    f_x2 = 2.0 * rashba_str * sin( M_PI / 6.0 ) / 3.0;


  std::complex<r_type> ImUnit(0,1);

  //  f_x*=ImUnit;
  // f_x2*=ImUnit;

  f_y *= ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){
      int n = C*W + j * W + i;


        ket[ 2 * n ]     = 0;
	ket[ 2 * n + 1 ] = 0;

	
      if( i!=0 ){
	ket[ 2 * n ]     += d_x2 * std::complex<r_type> (((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n - 2 ] + ( f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 1 ] );
	ket[ 2 * n + 1 ] += d_x2 * std::complex<r_type>(((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n - 1] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1)  + f_y ) * p_ket[ 2 * n - 2 ]  );
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     += d_x2 * std::complex<r_type>(((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n + 2 ] + ( f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 3 ]   );
	ket[ 2 * n + 1 ] += d_x2 * std::complex<r_type>(((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 2 ]   );
      }
      if(j != (Le-1)){
	ket[ 2 * n ]     +=   std::complex<r_type>((j+i)%2) * d_x * ( t * p_ket[ 2 * n + 2 * W ]    -  f_x * p_ket[ 2 * n + 2 * W + 1 ]) ;
	ket[ 2 * n + 1 ] +=   std::complex<r_type>((j+i)%2) * d_x * ( t * p_ket[ 2 * n + 2 * W + 1 ]  +  f_x * p_ket[ 2 * n + 2 * W ]);
      }
      if(j != 0){
	ket[ 2 * n ]     +=  - std::complex<r_type>((j+i+1)%2) * d_x * (  t * p_ket[ 2 * n - 2 * W ]   +  f_x * p_ket[ 2 * n - 2 * W + 1 ]);
	ket[ 2 * n + 1 ] +=  - std::complex<r_type>((j+i+1)%2) * d_x * (  t * p_ket[ 2 * n - 2 * W + 1 ] -  f_x * p_ket[ 2 * n - 2 * W ] );
      }
      

    }
 } 



 bool vertical_cbc=false;
 bool horizontal_cbc=false;

 if( (W/2)%2 !=0 && vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;
 
 if(vertical_cbc)
  for(int j=0; j<Le; j++){

    int n_up = j * W + W-1;
    int n_down = j * W;


    ket[ 2 * n_up ]     +=   d_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n_down + 1 ] );
    ket[ 2 * n_up + 1 ] +=   d_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) * ( -t * p_ket[ 2 * n_down + 1 ] + ( f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1)  - f_y ) * p_ket[ 2 * n_down ]  );
      
      
    ket[ 2 * n_down ]     +=  -d_x2 * std::complex<r_type>(((j+W)%2)==0? 1:-1) * ( -t * p_ket[ 2 * n_up ]     + ( -f_x2 * std::complex<r_type>(((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up + 1 ]   );
    ket[ 2 * n_down + 1 ] +=  -d_x2 * std::complex<r_type>(((j+W)%2)==0? 1:-1) * ( -t * p_ket[ 2 * n_up + 1 ] + (  f_x2 * std::complex<r_type>(((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up ]   );
      
 } 


 if(Le%2!=0&&horizontal_cbc)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;
 
 if(horizontal_cbc)
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (Le-1) * W + i;

      
        ket[ 2 * n_front ]     +=  - d_x * std::complex<r_type>((n_front+1)%2) * ( -t * p_ket[ 2 * n_back ]     + ( f_x * std::complex<r_type>(((n_front)%2)==0? 1:-1)  ) * p_ket[ 2 * n_back + 1 ] );
	ket[ 2 * n_front + 1 ] +=  - d_x * std::complex<r_type>((n_front+1)%2) * ( -t * p_ket[ 2 * n_back + 1 ] + ( -f_x * std::complex<r_type>(((n_front)%2)==0? 1:-1)  ) * p_ket[ 2 * n_back ] );


	ket[ 2 * n_back ]     +=  d_x * std::complex<r_type>((n_front)%2) * ( -t * p_ket[ 2 * n_front ] + ( -f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1) ) * p_ket[ 2 * n_front + 1 ]   );
	ket[ 2 * n_back + 1 ] +=  d_x * std::complex<r_type>((n_front)%2) * ( -t * p_ket[ 2 * n_front + 1 ] + ( f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1)  ) * p_ket[ 2 * n_front ]   );



    }




 
}; 



void ArmchairGraph_RashbaSOC::vel_op_y ( type* ket, type* p_ket){

  int Le = this->parameters().LE_,
      W      = this->parameters().W_,
      C      = this->parameters().C_;


  
  r_type t = t_standard_  ;
 
  if(this->parameters().C_%2==1)
    std::cout<<"BEWARE VX OP ONLY WORKS FOR EVEN C!!!!!!!!!!"<<std::endl;


      
  std::complex<r_type>
    rashba_str = rashba_str_ * t,
    d_y2 = a0_ * cos( M_PI / 6.0 ),
    f_y  = 2.0 * rashba_str * cos( M_PI / 6.0 ) / 3.0,
    f_x2 = 2.0 * rashba_str * sin( M_PI / 6.0 ) / 3.0;


  std::complex<r_type> ImUnit(0,1);

  //  f_x*=ImUnit;
  // f_x2*=ImUnit;

  f_y *= ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<Le; j++){
    for(int i=0; i<W; i++){
      int n = C*W + j * W + i;
	
      ket[ 2 * n ]     = 0;
      ket[ 2 * n + 1 ] = 0;
        
      if( i!=0 ){
	ket[ 2 * n ]     += d_y2 * std::complex<r_type> (((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n - 2 ] + ( f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 1 ] );
	ket[ 2 * n + 1 ] += d_y2 * std::complex<r_type>(((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n - 1] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1)  + f_y ) * p_ket[ 2 * n - 2 ]  );
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     += d_y2 * std::complex<r_type>(((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n + 2 ] + ( f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 3 ]   );
	ket[ 2 * n + 1 ] += d_y2 * std::complex<r_type>(((j+i)%2)==0? -1:1) * ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 2 ]   );
      }
    }
 } 



 bool vertical_cbc=false;

 if( (W/2)%2 !=0 && vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;
 
 if(vertical_cbc)
  for(int j=0; j<Le; j++){

    int n_up = j * W + W-1;
    int n_down = j * W;
      
    ket[ 2 * n_down ]     +=  -d_y2 * std::complex<r_type>(((j+W)%2)==0? 1:-1) * ( -t * p_ket[ 2 * n_up ]     + ( -f_x2 * std::complex<r_type>(((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up + 1 ]   );
    ket[ 2 * n_down + 1 ] +=  -d_y2 * std::complex<r_type>(((j+W)%2)==0? 1:-1) * ( -t * p_ket[ 2 * n_up + 1 ] + (  f_x2 * std::complex<r_type>(((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up ]   );
      
 } 
 
}; 



void ArmchairGraph_RashbaSOC::H_ket (r_type a, r_type b, type* ket, type* p_ket){

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
  

  //  f_x*=ImUnit;
  // f_x2*=ImUnit;

  std::complex<r_type> ImUnit(0,1);
  f_y *= ImUnit;
 
#pragma omp parallel for 
 for(int j=0; j<fullLe; j++){
    for(int i=0; i<W; i++){
      int n = j * W + i;

      ket[ 2 * n ]     = ( b_a - m_str ) * p_ket[ 2 * n ] ;
      ket[ 2 * n + 1 ] = ( b_a + m_str ) * p_ket[ 2 * n + 1] ;

      
      if( i!=0 ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2 * n - 2 ] + (  f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 1 ] );
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n - 1 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n - 2 ] );
      }
      if(i != (W-1) ){
	ket[ 2 * n ]     +=  ( -t * p_ket[ 2 * n + 2 ] + (  f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 3 ]   );
	ket[ 2 * n + 1 ] +=  ( -t * p_ket[ 2 * n + 3 ] + ( -f_x2 * std::complex<r_type>(((j+i)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n + 2 ]   );
      }
      if(j != (fullLe-1)){
	ket[ 2 * n ]     +=  std::complex<r_type> ((j+i)%2) * ( -t * p_ket[ 2*n+2*W ]           +  f_x * p_ket[ 2 * n + 2 * W + 1 ] ) ;
	ket[ 2 * n + 1 ] +=  std::complex<r_type> ((j+i)%2) * ( -t * p_ket[ 2 * n + 2 * W + 1]  -  f_x * p_ket[ 2 * n + 2 * W ]);
      }
      if(j != 0){
	ket[ 2 * n ]     += std::complex<r_type>((j+i+1)%2) * ( - t * p_ket[ 2 * n - 2 * W ]     -  f_x * p_ket[ 2 * n - 2 * W + 1 ] );
	ket[ 2 * n + 1 ] += std::complex<r_type>((j+i+1)%2) * ( - t * p_ket[ 2 * n - 2 * W + 1 ] +  f_x * p_ket[ 2 * n - 2 * W ]   );
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



 
 bool vertical_cbc=false;
 bool horizontal_cbc=false;


  if(W%2!=0&&vertical_cbc)
   std::cout<<"BEWARE VERTICAL CBC ONLY WORKS FOR EVEN M!!!!"<<std::endl;

  
 if(vertical_cbc)
  for(int j=0; j<fullLe; j++){

    int n_up = j * W + W - 1;
    int n_down = j * W;


    ket[ 2 * n_up ]    +=  sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down ]    + ( -f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n_down + 1] );
    ket[ 2 * n_up + 1] +=  sFilter[ n_up ] * ( -t * p_ket[ 2 * n_down + 1] + (  f_x2 * std::complex<r_type>(((j)%2)==0? 1:-1) - f_y ) * p_ket[ 2 * n_down ]  );
      
      
    ket[ 2 * n_down]      +=  sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up ] + ( -f_x2 * std::complex<r_type>( ((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up + 1 ]   );
    ket[ 2 * n_down + 1 ] +=  sFilter[ n_down ] * ( -t * p_ket[ 2 * n_up + 1 ] + ( f_x2 * std::complex<r_type>(((j+W-1)%2)==0? 1:-1) + f_y ) * p_ket[ 2 * n_up ]   );
      
 } 


 if(fullLe%2!=0&&horizontal_cbc)
   std::cout<<"BEWARE HORIZONTAL CBC ONLY WORKS FOR EVEN LE+C!!!!"<<std::endl;

 
 if(horizontal_cbc)
    for(int i=0; i<W; i++){
      int n_front = i;
      int n_back = (fullLe-1) * W + i;

      
	ket[ 2 * n_front ]     +=  sFilter[ n_front ] * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_back ] + ( -f_x * std::complex<r_type>(((n_front)%2)==0? 1:-1) ) * p_ket[ 2 * n_back + 1 ] );
	ket[ 2 * n_front + 1 ] +=  sFilter[ n_front ] * ((n_front+1)%2) *( -t * p_ket[ 2 * n_back + 1 ] + ( f_x * std::complex<r_type>(((n_front)%2)==0? 1:-1)  ) * p_ket[ 2 * n_back ]  );


	ket[ 2 * n_back ]     +=  sFilter[ n_back ] * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_front ] + ( f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1)  ) * p_ket[ 2 * n_front + 1 ]   );
	ket[ 2 * n_back + 1 ] +=  sFilter[ n_back ] * ((n_front+1)%2) * ( -t * p_ket[ 2 * n_front + 1 ] + ( -f_x * std::complex<r_type>(((n_back)%2)==0? 1:-1)  ) * p_ket[ 2 * n_front ]   );

    }


 
}; 





  /*

      
      if( i!=0 ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n-2) + ( f_x2 * (((j+i)%2)==0? 1:-1) - f_y ) * p_ket(2*n-1) );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n-1) + ( -f_x2 * (((j+i)%2)==0? 1:-1)  - f_y ) * p_ket(2*n-2)  );
      }
      if(i != (M-1) ){
	ket(2*n)   += 2 * ( -t * p_ket(2*n+2) + ( f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n+3)   );
	ket(2*n+1) += 2 * ( -t * p_ket(2*n+3) + ( -f_x2 * (((j+i)%2)==0? 1:-1) + f_y ) * p_ket(2*n+2)   );
      }
      if(j != (fullLe-1)){
	ket(2*n)   += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M)    +  f_x * p_ket(2*n+2*M+1)) ;
	ket(2*n+1) += 2 * ((j+i)%2) * ( -t * p_ket(2*n+2*M+1)  -  f_x * p_ket(2*n+2*M));
      }
      if(j != 0){
	ket(2*n)   += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M)   -  f_x * p_ket(2*n-2*M+1) );
	ket(2*n+1) += 2 * ((j+i+1)%2) * ( - t * p_ket(2*n-2*M+1) +  f_x * p_ket(2*n-2*M)   );
      }
      

*/




