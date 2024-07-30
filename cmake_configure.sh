rm -rf build; mkdir build;
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${PWD}
popd
