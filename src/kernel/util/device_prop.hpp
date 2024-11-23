#pragma once

class __attribute__((visibility("default"))) DeviceProp {
public:
    int warpsize;
    int major, minor;
    int multiprocessorCount;
    int maxSharedMemPerBlock;

    DeviceProp(int device_id);
};