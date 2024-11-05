#. env.sh

rm -rf build; mkdir build;
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PWD} \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
popd
