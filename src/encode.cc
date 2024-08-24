#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <bitset>
#include <deque>

#include "AudioFile.h"
#include <dlib/svm.h>


using namespace std;
using namespace dlib;



std::vector<uint8_t> read_wav_header(std::istream& inputStream) {
    std::vector<uint8_t> header;

    // Read the header
    header.resize(44);
    inputStream.read(reinterpret_cast<char*>(header.data()), header.size());

    return header;
}

int16_t neuralink_10bit_to_16bit(uint16_t u) {
    double temp = (u - 512 + 0.5) * (64.0 + 1009.0 / 16384.0) - 0.5;
    return static_cast<int16_t>(std::trunc(temp));
}

uint16_t neuralink_16bit_to_10bit(int16_t x) {
    return (x >> 6) + 512;
}

template<class T>
void displayBinary(T value) {
    // Convert the integer to a 32-bit binary representation
    std::bitset<16> binary(value);
    std::cout << binary << " v=" << value << std::endl;
}

bool neuralink_read_symbol_from_stream(std::istream &inputStream, int16_t &symbol) {
    float raw_sample;
    // if (inputStream.read(reinterpret_cast<char*>(&raw_sample), sizeof(raw_sample))) {
    //     // symbol = neuralink_16bit_to_10bit(raw_sample);
    //     symbol = raw_sample;
    //     return true;
    // }
    inputStream >> symbol;
    std::cout << "sym " << std::fixed << std::setw(11) << std::setprecision(6) 
          << std::setfill('0') << symbol << endl;
    return true;
}

int main( int argc, char** argv) {
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    std::stringstream sstr;
    sstr << "cp " << inputPath << " " << outputPath;
    system(sstr.str().c_str());

    AudioFile<int16_t> audioFile;
    audioFile.load (inputPath);

    int sampleRate = audioFile.getSampleRate();
    int bitDepth = audioFile.getBitDepth();

    int numSamples = audioFile.getNumSamplesPerChannel();
    double lengthInSeconds = audioFile.getLengthInSeconds();

    int numChannels = audioFile.getNumChannels();
    bool isMono = audioFile.isMono();
    bool isStereo = audioFile.isStereo();

    // or, just use this quick shortcut to print a summary to the console
    audioFile.printSummary();
    // return 0;

    int channel = 0;
    // int numSamples = audioFile.getNumSamplesPerChannel();

    uint16_t lastValue = 0;
    int16_t curDelta = 0;

    int16_t maxDelta = 0;
    const int maxWindow = 5;
    std::deque<uint16_t> dodWindow;

    ofstream ofs("dvalues.csv", std::ios::app);

    // Learning
    // Here we declare that our samples will be 1 dimensional column vectors.  
    typedef matrix<double,maxWindow,1> sample_type;
    sample_type m;
    std::vector<sample_type> samples;
    std::vector<double> labels;
    uint16_t lastSample = 0;
    for (int i = 0; i < numSamples; i++)
    {
        int16_t currentSample = audioFile.samples[channel][i];
        uint16_t isample = neuralink_16bit_to_10bit(currentSample);
        if (dodWindow.size() >= maxWindow)
        {
            for( int k = 0; k < maxWindow; ++k )
            {
                m(k) = dodWindow[k];
            }
            samples.push_back(m);
            labels.push_back(isample);
        }
        dodWindow.push_back(isample);
        while( dodWindow.size() > maxWindow )
            dodWindow.pop_front();
    }
    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef radial_basis_kernel<sample_type> kernel_type;

    // Here we declare an instance of the krr_trainer object.  This is the
    // object that we will later use to do the training.
    krr_trainer<kernel_type> trainer;

    // Here we set the kernel we want to use for training.   The radial_basis_kernel 
    // has a parameter called gamma that we need to determine.  As a rule of thumb, a good 
    // gamma to try is 1.0/(mean squared distance between your sample points).  So 
    // below we are using a similar value computed from at most 2000 randomly selected
    // samples.
    const double gamma = 5.0/compute_mean_squared_distance(randomly_subsample(samples, 2000));
    cout << "using gamma of " << gamma << endl;
    trainer.set_kernel(kernel_type(gamma));

    // now train a function based on our sample points
    decision_function<kernel_type> test = trainer.train(samples, labels);
    //

    // 
    // return 0;

    // 
    for (int i = 0; i < numSamples; i++)
    {
        int16_t currentSample = audioFile.samples[channel][i];
        uint16_t isample = neuralink_16bit_to_10bit(currentSample);

        if (dodWindow.size() >= maxWindow)
        {
            for( int k = 0; k < maxWindow; ++k )
            {
                m(k) = dodWindow[k];
            }
            auto prediction = test(m);
            int16_t d = ( (int16_t)isample - (int16_t)prediction );
            cerr << "actual=" << isample << " pred=" << prediction << " d=" << d << endl;
            if( abs(d) > abs(maxDelta) ){
                maxDelta = d;
            }
            ofs << d << endl;
        }

        if( dodWindow.size() >= 2 ){
            int16_t t_n = (int16_t)isample;
            int16_t t_n_1 = (int16_t)dodWindow[1];
            int16_t t_n_2 = (int16_t)dodWindow[0];
            // 
            int16_t d = ( t_n - t_n_1 );
            int16_t D = ( t_n - t_n_1 ) - ( t_n_1 - t_n_2 );
            // std::cerr << "t_n=" << isample << " t_n_1=" << t_n_1 << " t_n_2=" << t_n_2 << " D=" << D << " d=" << d << endl;
            // ofs << d << endl;

            // // Looking for min_d
            // uint16_t min_ad = 7777;
            // int16_t min_d = 0;
            // uint16_t min_d_pos = 0;
            // for( int j = 0; j < dodWindow.size(); ++j ) {
            //     int16_t d = t_n - dodWindow[j];
            //     uint16_t ad = abs( d );
            //     if( ad < min_ad )
            //     {
            //         min_ad = ad;
            //         min_d = d;
            //         min_d_pos = j;
            //     }
            // }
            // std::cerr << "ref_d=" << d << " min_d=" << min_d << " min_d_pos=" << min_d_pos << endl;
            
        }

        // int16_t isample2 = neuralink_10bit_to_16bit(isample);
        // // memcpy( &isample, &currentSample, sizeof(currentSample) );
        // // std::cerr << "f=" << currentSample << " i=" << isample << endl;
        // // displayBinary( currentSample );
        // // displayBinary( isample );
        // // displayBinary( isample2 );
        // int16_t delta = (int16_t)isample - (int16_t)lastValue;
        // int16_t deltaOfDelta = delta - curDelta;
        // curDelta = delta;
        // std::cerr << "value=" << isample << " delta=" << delta << " delta_of_delta=" << deltaOfDelta << endl;
        // uint16_t absDelta = abs(deltaOfDelta);
        // if ( i != 0 && absDelta > maxDelta ) {
        //     // cerr << "last=" << lastValue << " value=" << isample << " i=" << i << endl;
        //     maxDelta = absDelta; 
        //     displayBinary( abs(delta) );
        // }
        // lastValue = isample;
        // cerr << "*" << endl;
        // displayBinary( abs(delta) );
        // std::cerr << "-" << endl;

        dodWindow.push_back(isample);
        while( dodWindow.size() > maxWindow )
            dodWindow.pop_front();

    }
    cerr << "maxDelta=" << maxDelta << endl;

    return 0;
}