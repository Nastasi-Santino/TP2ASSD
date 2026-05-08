#include "fft.hpp"
#include <complex>  // std::complex, std::polar
#include <cstdlib>  // malloc, free
#include <cstddef>  // size_t
#include <cstdio>   // printf
#include <chrono>   // timing
#define N_s 32768  // 2^15
#define PI 3.14159265358979323846f

void genData(std::complex<float> * data, uint16_t n){
    for(unsigned int count=0; count < n; count++){
        data[count] = std::complex<float>(10*cosf(2.0f * PI * count / n), 0.0f);
        //data[count] = std::complex<float>((count + 1)*10.0f, 0.0f);
    }
}

void dft(std::complex<float> *in , std::complex<float> *out , size_t N){
    std::complex<float> * output = (std::complex<float> *) malloc(sizeof(std::complex<float>)*N);
        for(size_t k = 0; k < N; k++){
            output[k] = 0.0f;
            for(size_t n = 0; n < N; n++){
                output[k] += in[n] * std::polar(1.0f, -2.0f*PI*k*n/N);
            }
        }
        for(size_t k = 0; k < N; k++){
            out[k] = output[k];
        }
    free(output);
}


int main(){
    uint16_t n = N_s;

    fft myFFT(n);

    std::complex<float> input[N_s];
    genData(input, n);

    std::complex<float> outputFFT[N_s];
    std::complex<float> outputDFT[N_s];

    // Time FFT
    auto fft_start = std::chrono::high_resolution_clock::now();
    myFFT.computeFFT(input, outputFFT);
    auto fft_end = std::chrono::high_resolution_clock::now();
    double fft_time_ms = std::chrono::duration<double, std::milli>(fft_end - fft_start).count();

    myFFT.~fft();

    genData(input, n);
    
    // Time DFT
    auto dft_start = std::chrono::high_resolution_clock::now();
    dft(input, outputDFT, n);
    auto dft_end = std::chrono::high_resolution_clock::now();
    double dft_time_ms = std::chrono::duration<double, std::milli>(dft_end - dft_start).count();


    float error_real = 0.0f;
    float error_imag = 0.0f;
    for(unsigned int count=0; count < N_s; count++){
        error_real += outputFFT[count].real() - outputDFT[count].real();
        error_imag += outputFFT[count].imag() - outputDFT[count].imag();
    }

    float error_magnitude = sqrt(error_real * error_real + error_imag * error_imag);
    printf("Error Magnitude: %f\n", error_magnitude);

    // Export results to CSV
    FILE* csv_file = fopen("fft_dft_comparison.csv", "w");
    fprintf(csv_file, "Index,FFT_Real,FFT_Imag,DFT_Real,DFT_Imag\n");
    for(unsigned int i = 0; i < n; i++){
        fprintf(csv_file, "%d,%f,%f,%f,%f\n", 
            i, 
            outputFFT[i].real(), outputFFT[i].imag(),
            outputDFT[i].real(), outputDFT[i].imag());
    }
    fclose(csv_file);
    printf("\nResults exported to fft_dft_comparison.csv\n");

    // Print timing results
    printf("\n=== TIMING RESULTS (N=%d) ===\n", N_s);
    printf("FFT Time: %.6f ms\n", fft_time_ms);
    printf("DFT Time: %.6f ms\n", dft_time_ms);
    printf("Speedup: %.2fx\n", dft_time_ms / fft_time_ms);

    return 0;

}