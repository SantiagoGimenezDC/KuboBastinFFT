#!/bin/bash

#Giant 24 million atom example of Graphene, with very high resolution (M=10000). 
#Should use approximately 320GB memory, and take less than 30hrs to complete.
#Runs best in 16 cores

OMP_NUM_THREADS=16 ./../../KuboBastinFFT SimData.dat
