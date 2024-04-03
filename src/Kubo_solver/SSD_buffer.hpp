#ifndef SSD_BUFFER_HPP
#define SSD_BUFFER_HPP

#include<iostream>
#include<fstream>
#include<string>

#include<string>
#include"../static_vars.hpp"
#include<eigen-3.4.0/Eigen/Core>


class SSD_buffer{
private:
  int COLS_, ROWS_, ROWS_stride_, COLS_stride_, ROWS_rest_, COLS_rest_, num_buffers_;
  double  SSD_size_, RAM_size_;
  std::string filename_;
  bool rest_buffer_=false;

  FILE* file_;

  FILE* out_, *in_;
public:
  ~SSD_buffer(){
    //    file_ = fopen( filename_.c_str(), "wb"); //hack to overwrite filename_ file with empty file;
    //fclose(file_);
  };
  
  int COLS_stride(){return COLS_stride_;};
  int ROWS_stride(){return ROWS_stride_;};
  int num_buffers(){return num_buffers_;};

  int COLS_rest(){return COLS_rest_;};
  
  SSD_buffer(int COLS, int ROWS, double RAM_size, std::string filename ) : COLS_(COLS), ROWS_(ROWS), RAM_size_(RAM_size), filename_(filename){

    SSD_size_ = 2 * double(COLS_) * double(ROWS_) * sizeof(type);
    
    num_buffers_ = int(std::ceil( SSD_size_ / (RAM_size_))  );

    unsigned long long int SSD_size_int = static_cast<long int>(SSD_size_),
    RAM_size_int = static_cast<long int>(RAM_size_);
    if(num_buffers_==0){
     num_buffers_ = 1;
     ROWS_stride_ = ROWS_  / num_buffers_ ;
     COLS_stride_ = COLS_  / num_buffers_ ;
     ROWS_rest_   = 0;
     COLS_rest_   = 0;
   }
    else if( SSD_size_int % RAM_size_int == 0){
     ROWS_stride_ = ROWS_  / num_buffers_ ;
     COLS_stride_ = COLS_  / num_buffers_ ;
     ROWS_rest_   = 0;
     COLS_rest_   = 0;
   }
   else if( SSD_size_int % RAM_size_int  > ROWS_ ){ //This will assume ROWS_>COLS_ always;
     while( (COLS_  % num_buffers_ ) > (COLS_ / num_buffers_) )
       num_buffers_++;

     ROWS_stride_ = ROWS_  / ( num_buffers_  ) ;
     COLS_stride_ = COLS_  / ( num_buffers_  ) ;
     ROWS_rest_   = ROWS_  % ( num_buffers_  ) ;
     COLS_rest_   = COLS_  % ( num_buffers_ ) ;
     rest_buffer_ = true;
   }
   else{

     std::cout<<"I'm taking the liberty of increasing RAM buffer juust a little bit cmooon."<<std::endl;
     RAM_size_+=static_cast<double>(SSD_size_int % RAM_size_int);

     ROWS_stride_ = ROWS_  / num_buffers_ ;
     COLS_stride_ = COLS_  / num_buffers_ ;
     ROWS_rest_ =0;
     COLS_rest_ =0;
   }
   
   std::cout<<"Ratio between SSD/RAM:  "<<num_buffers_<<"  SSD buffer size: "<< SSD_size_/1000000<<"MB"<<std::endl;



  file_ = fopen( filename_.c_str(), "wb");
  fclose(file_);

  };


  void reset_buffer(){
    out_ = fopen( filename_.c_str(), "wb");
    fclose(out_);
  }
  
  void begin_upload(){
    out_ = fopen( filename_.c_str(), "a+");
  }
  
  void upload_col_buffer_to_SSD(int buffer_num, type* RAM_buffer){

    //  FILE* out = fopen( filename_.c_str(), "a+");

    if(buffer_num < num_buffers_)
      fwrite( RAM_buffer, 1, COLS_stride_ * ROWS_ * sizeof(type), out_ );    
    else if(buffer_num == num_buffers_){
      if(rest_buffer_)
        fwrite( RAM_buffer, 1, COLS_rest_ * ROWS_ * sizeof(type), out_ );
      else
	fwrite( RAM_buffer, 1, COLS_stride_ * ROWS_ * sizeof(type), out_ );
    }
    else
      std::cout<<"  Wrong buffer_num "<<buffer_num<<std::endl;

    //fclose(out);
    
  }
  
  void end_upload(){
    fclose(out_);
  }

  
  
  int retrieve_row_buffer_from_SSD(int buffer_num, type RAM_buffer[] ){
    int buffer_size = ROWS_stride_;
    
    if(rest_buffer_==true && (buffer_num == (num_buffers_ ) ) ){
     buffer_size = ROWS_rest_;
    }


    FILE* in = fopen(filename_.c_str(), "rb");
   

    for(int j = 0; j < COLS_; j++){
      //     std::cout<<j<<"  "<<buffer_num<<"  "<<buffer_size<<"  "<<ROWS_stride_<<"  "<<ROWS_<<"  "<<COLS_<<filename_<<std::endl;
      fseek(in, (  j * ROWS_ + buffer_num * ROWS_stride_ ) * sizeof(type), SEEK_SET);
      //fseek(in, (  ROWS_   ) * sizeof(type), SEEK_CUR);
      fread( &RAM_buffer[ j * buffer_size], 1, buffer_size  * sizeof(type), in );
    }
    
    fclose(in);
 
    return buffer_size;
  }
  
};


#endif //SSD_BUFFER_HPP
