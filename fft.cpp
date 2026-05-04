#include "fft.hpp"

#define M_PI 3.14159265358979323846

fft::fft(uint16_t n){
    this->points = n;
    this->dualNodeDistance = n/2;
    this->stage = 1;
    this->maxStage = log2(n);
    this->group = 1;
    this->maxGroup = 1;
    this->butterflysCount = 0;
    this->maxButterflys = n/2;
    this->distanceToNextGroup = n;

    this->Coeff = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
    this->bitReverse_LUT = (uint16_t *)malloc(n * sizeof(uint16_t));
    generateTwiddleFactors();
    generateBitReverseLUT();

}

void fft::generateTwiddleFactors(){

    float freq = 2 * M_PI / this->points;
    float angle = 0.0;

    for(unsigned int count=0; count < this->points/2; count++){
        angle = freq * count;
        this->Coeff[count] = std::complex<float>(cos(angle), -sin(angle));
    }   
}

void fft::generateBitReverseLUT(void){

    for(unsigned int count=0; count < this->points; count++){
        uint16_t reversed = 0;
        uint16_t temp = count;

        for(unsigned int bit=0; bit < log2(this->points); bit++){
            reversed <<= 1;
            reversed |= (temp & 1);
            temp >>= 1;
        }
        this->bitReverse_LUT[count] = reversed;
    }

}

void fft::butterfly(std::complex<float> * pIn, std::complex<float> * pOut, std::complex<float> * pCoeff){

    float a_in_real = pIn->real();
    float a_in_imag = pIn->imag();

    float b_in_real = (pIn + this->dualNodeDistance)->real();
    float b_in_imag = (pIn + this->dualNodeDistance)->imag();

    float c_real = pCoeff->real();
    float c_imag = pCoeff->imag();

    float a_out_real = a_in_real - c_real * b_in_real - c_imag * b_in_imag;
    float a_out_imag = a_in_imag - c_real * b_in_imag + c_imag * b_in_real;

    float b_out_real = 2 * a_in_real - a_out_real;
    float b_out_imag = 2 * a_in_imag - a_out_imag;

    *pOut = std::complex<float>(a_out_real, a_out_imag);
    *(pOut + this->dualNodeDistance) = std::complex<float>(b_out_real, b_out_imag);

}

void fft::swap(std::complex<float> * a, std::complex<float> * b){
    std::complex<float> temp = *a;
    *a = *b;
    *b = temp;
}

void fft::unascramble(std::complex<float> * pData){
    for(unsigned int count=0; count < this->points; count++){
        uint16_t reversedIndex = this->bitReverse_LUT[count];
        if(count < reversedIndex){
            swap(pData + count, pData + reversedIndex);
        }
    }
}

void fft::computeFFT(std::complex<float> * in, std::complex<float> * out){

    uint16_t index;
    std::complex<float> * pData;
    std::complex<float> * pOut;
    std::complex<float> * pCoeff = this->Coeff;
    for(stage = 1; stage <= maxStage; stage++){

        if(stage == 1){
            pData = in;
        } else{
            pData = out;
        }
        pCoeff = this->Coeff;
        pOut = out;

        for(group = 1; group <= maxGroup; ){

            for(butterflysCount = 1; butterflysCount <= maxButterflys; butterflysCount++){
                butterfly(pData, pOut, pCoeff);
                pData++;
                pOut++;
            }

            group++;

            index = this->bitReverse_LUT[2*(group-1)];
            pCoeff = &this->Coeff[index];       
            pData = out + distanceToNextGroup * (group-1); // stage == 1 irrelevant because only 1 group

        }
    
        maxGroup *= 2;
        distanceToNextGroup /= 2;
        maxButterflys /= 2;
        dualNodeDistance /= 2;
    }

    unascramble(out);
    this->stage = 1;
    this->group = 1;
    this->butterflysCount = 0;
    this->maxGroup = 1;
    this->maxButterflys = this->points / 2;
    this->dualNodeDistance = this->points / 2;
    this->distanceToNextGroup = this->points;

}