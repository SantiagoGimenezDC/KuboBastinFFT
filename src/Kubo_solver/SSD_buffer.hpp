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
  int COLS_, ROWS_, ROWS_stride_, COLS_stride_, ROWS_rest_, COLS_rest_, num_write_buffers_, num_read_buffers_;
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
  int num_buffers(){return num_read_buffers_;};

  int COLS_rest(){return COLS_rest_;};
  
  SSD_buffer(int COLS, int ROWS, double RAM_size, std::string filename ) : COLS_(COLS), ROWS_(ROWS), RAM_size_(RAM_size), filename_(filename){

    SSD_size_ =  double(COLS_) * double(ROWS_) * sizeof(type);
    
    std::size_t SSD_size_int = static_cast<size_t>(SSD_size_),
                RAM_size_int = static_cast<size_t>(RAM_size_);


    
    ROWS_stride_ = RAM_size_int >= SSD_size_int ? ROWS_ : RAM_size_int /( sizeof(type) * COLS_) ;
    COLS_stride_ = RAM_size_int >= SSD_size_int ? COLS_ : RAM_size_int /( sizeof(type) * ROWS_);
      
    num_write_buffers_ = static_cast<int>( COLS_ / COLS_stride_  );
    num_read_buffers_  = static_cast<int>( ROWS_ / ROWS_stride_  );

    ROWS_rest_   = ROWS_  %  num_read_buffers_  ;
    COLS_rest_   = COLS_  %  num_write_buffers_ ;


    std::cout<<COLS_stride_<<" "<<COLS_rest_<<"      "<<ROWS_stride_<<" "<<ROWS_rest_<<std::endl;
    //    while( (COLS_ % num_buffers_ ) > (COLS_ / num_buffers_) || (ROWS_ % num_buffers_ ) > (ROWS_ / num_buffers_) )
    //  num_buffers_++;
      
      //if( ROWS_>COLS_ && SSD_size_int % RAM_size_int  > ROWS_ );

      
    std::cout<<"RAM buffer size:  "<<RAM_size_/1E9 <<"GBs   SSD buffer size: "<< SSD_size_/1E9<<"GBs"<<std::endl;
    std::cout<<"Number of SSD writes:  "<<num_write_buffers_ <<";  Number of SSD reads: "<< num_read_buffers_<<std::endl;

  };

  
  void upload_col_buffer_to_SSD(int buffer_num, type* RAM_buffer){
    
    std::size_t buffer_size = ( buffer_num == num_write_buffers_ ? COLS_rest_ : COLS_stride_);

    if(buffer_size == 0)
      return ;
    
    buffer_size *= ROWS_;

    if(buffer_num>num_write_buffers_){
      std::cout<<"invalid buffer num:  "<<buffer_num<<"/"<<num_write_buffers_<<std::endl;
      return;
    }
      /*
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fileDesc = open(filename_.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_APPEND | O_NONBLOCK, mode);
    std::cout<<write(fileDesc, (void*)RAM_buffer, buffer_size*sizeof(type))<<std::endl;
    */
    FILE* out = fopen( filename_.c_str(), "a+");
    
    fwrite( RAM_buffer, 1, buffer_size * sizeof(type), out );
      
    fclose(out);
    
  }
  

  
  
  int retrieve_row_buffer_from_SSD(int buffer_num, type RAM_buffer[] ){
    
    std::size_t buffer_size = ( buffer_num == num_read_buffers_ ? ROWS_rest_ : ROWS_stride_);

    if(buffer_size == 0)
      return 0;

    
    //    int in_f = open(filename_.c_str(), O_DIRECT | O_RDONLY);
    FILE* in = fopen(filename_.c_str(), "rb");


      
    for(int j = 0; j < COLS_; j++){
      
      fseek(in, (  j * ROWS_ + buffer_num * ROWS_stride_ ) * sizeof(type) , SEEK_SET);
      fread( &RAM_buffer[ j * buffer_size], 1, buffer_size  * sizeof(type), in );

    }
      fclose(in);    

    //fread( RAM_buffer, 1, COLS_ * buffer_size  * sizeof(type), in );
    
 
    return buffer_size;
  }
  
};


#endif //SSD_BUFFER_HPP
