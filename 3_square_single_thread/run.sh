rm -r build
mkdir build
cd build
cmake .. && make && ./bin/exe
cd ..
rm -r build