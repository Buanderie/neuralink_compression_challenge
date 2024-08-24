#include <iostream>
#include <sstream>

using namespace std;

int main( int argc, char** argv) {
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    std::stringstream sstr;
    sstr << "cp " << inputPath << " " << outputPath;
    system(sstr.str().c_str());

    
    return 0;
}