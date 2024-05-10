#include "add.cuh"
#include "clock.cuh"
#include <stdio.h>
#include <iostream>
#include <typeinfo>


int main()
{
    if(typeid(double) == typeid(real))
    {
        // std::cout << "using double precision version.\n" << std::endl;
        printf("using double precision version.\n");
    }

    cuda_clock();

    return 0;
}