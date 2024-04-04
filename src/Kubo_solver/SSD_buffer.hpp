#ifndef SSD_BUFFER_HPP
#define SSD_BUFFER_HPP

#include<iostream>
#include<fstream>
#include<string>

#include<string>
#include"../static_vars.hpp"
#include<eigen-3.4.0/Eigen/Core>
#include<fcntl.h>
#include<unistd.h>

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

    SSD_size_ =  double(COLS_) * double(ROWS_) * sizeof(type);
    
    num_buffers_ = int(std::ceil( SSD_size_ / (RAM_size_))  );

    //std::size_t SSD_size_int = static_cast<size_t>(SSD_size_),
    //RAM_size_int = static_cast<size_t>(RAM_size_);
    
    while( (COLS_ % num_buffers_ ) > (COLS_ / num_buffers_) || (ROWS_ % num_buffers_ ) > (ROWS_ / num_buffers_) )
        num_buffers_++;
      
      ROWS_stride_ = ROWS_  / num_buffers_ ;
      COLS_stride_ = COLS_  / num_buffers_ ;
      ROWS_rest_   = ROWS_  % ( num_buffers_  );
      COLS_rest_   = COLS_  % ( num_buffers_  );
      
      //if( ROWS_>COLS_ && SSD_size_int % RAM_size_int  > ROWS_ );

      
    std::cout<<"Ratio between SSD/RAM:  "<<num_buffers_ <<"  SSD buffer size: "<< SSD_size_/1E9<<"GB"<<std::endl;

  };


  void reset_buffer(){
    out_ = fopen( filename_.c_str(), "wb");
    fclose(out_);
  }
  
  void begin_upload(){
    int out_f = open(filename_.c_str(),  O_DIRECT | O_WRONLY);
    out_ = fdopen( out_f, "a");
  }
  
  void upload_col_buffer_to_SSD(int buffer_num, type* RAM_buffer){
    std::size_t buffer_size = COLS_stride_;
    bool rest = ! ( COLS_rest_ == 0 );

    
    
    if(buffer_num == num_buffers_){
      if(rest)
        buffer_size = COLS_rest_;
      else return;
    }
    
    buffer_size *= ROWS_;


    
    /*
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fileDesc = open(filename_.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_APPEND | O_NONBLOCK, mode);
    std::cout<<write(fileDesc, (void*)RAM_buffer, buffer_size*sizeof(type))<<std::endl;
    */
    //  FILE* out = fopen( filename_.c_str(), "a+");

    
    fwrite( RAM_buffer, 1, buffer_size * sizeof(type), out_ );
      


    //fclose(out);
    
  }
  
  void end_upload(){
    fclose(out_);
  }

  
  
  int retrieve_row_buffer_from_SSD(int buffer_num, type RAM_buffer[] ){
    
    std::size_t buffer_size = ROWS_stride_;
     bool rest = !( ROWS_rest_ == 0 );


    if(buffer_num == num_buffers_){
      if(rest)
        buffer_size = ROWS_rest_;
      else return 0;
      }



      int in_f = open(filename_.c_str(), O_DIRECT | O_RDONLY);
      FILE* in = fdopen(in_f, "rb");

      
    for(int j = 0; j < COLS_; j++){
      
      fseek(in, (  j * ROWS_ + buffer_num * ROWS_stride_ ) * sizeof(type), SEEK_SET);
      fread( &RAM_buffer[ j * buffer_size], 1, buffer_size  * sizeof(type), in );
      

    }
      fclose(in);    

    //fread( RAM_buffer, 1, COLS_ * buffer_size  * sizeof(type), in );
    
 
    return buffer_size;
  }
  
};


#endif //SSD_BUFFER_HPP
