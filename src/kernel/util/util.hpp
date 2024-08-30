#pragma once

// Taken from Stack Overflow
inline size_t round_up(size_t in, size_t multiple) {
    if (multiple == 0)
        return in;

    int remainder = in % multiple;
    if (remainder == 0)
        return in ;

    return in + multiple - remainder;
}