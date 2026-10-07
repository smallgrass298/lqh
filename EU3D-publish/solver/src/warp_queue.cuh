#pragma once
#include <cuda_runtime.h>

// Entire physical warp must call with the same bucket state. No lane exits early.
// Heads contain RELATIVE positions and are zeroed before each kernel launch.
// A claim never crosses a bucket boundary; each lane solves one independent cell.
// Heavy buckets are offered first. A warp moves on when no unclaimed work remains,
// without waiting for other warps still computing their previously claimed chunks.
__device__ __forceinline__ bool warp_queue_claim(
    const int* offsets, int* heads, int& bucket, int& begin, int& end) {
    while (bucket >= 0) {
        const int size=offsets[bucket+1]-offsets[bucket];
        if (size) {
            int position=0;
            if ((threadIdx.x&31)==0) position=atomicAdd(heads+bucket,32);
            position=__shfl_sync(0xffffffffu,position,0);
            if (position<size) {
                begin=offsets[bucket]+position;
                end=begin+min(32,size-position);
                return true;
            }
        }
        --bucket;
    }
    return false;
}
