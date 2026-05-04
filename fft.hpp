#include <complex>
#include <cstdlib>
#include <cmath>

class fft{
    public: 
        fft(uint16_t n);
        ~fft(void){
            free(this->Coeff);
            free(this->bitReverse_LUT);
        }

        void computeFFT(std::complex<float> * in, std::complex<float> * out);


    private:
        uint16_t points;
        uint16_t dualNodeDistance;
        uint16_t stage;
        uint16_t maxStage;
        uint16_t group;
        uint16_t maxGroup;
        uint16_t butterflysCount;
        uint16_t maxButterflys;
        uint16_t distanceToNextGroup;

        std::complex<float> * Coeff;
        uint16_t * bitReverse_LUT;

        void generateTwiddleFactors(void);
        void generateBitReverseLUT(void);
        void butterfly(std::complex<float> * pIn, std::complex<float> * pOut, std::complex<float> * pCoeff);
        void swap(std::complex<float> * a, std::complex<float> * b);
        void unascramble(std::complex<float> * pData);
};