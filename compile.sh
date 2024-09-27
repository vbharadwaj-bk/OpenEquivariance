# If SOURCED_ENV_SCRIPT is not set, run env.sh 

if [ -z ${SOURCED_ENV_SCRIPT+x} ]; then
    echo "Sourcing env.sh."
    . env.sh
fi

pushd build
make -j 4
make install
popd