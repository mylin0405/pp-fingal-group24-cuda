# pp-fingal-group24-cuda
Author: Ming-Yu Lin & Ying-Hao Wan.   
Cuda implementation of parallel gaussian process
## Usage
```
mkdir build
cmake -B build/ -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build/ --config Release
./bin/app < data/mnist_2000_200.txt
```
