#include "warp_queue.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void visit(const int* offsets,int* heads,const int* list,int* seen,int* bad) {
    int bucket=7,begin=0,end=0;
    while(warp_queue_claim(offsets,heads,bucket,begin,end)) {
        int q=begin+(threadIdx.x&31);
        if(q<end) {
            if(q<offsets[bucket] || q>=offsets[bucket+1]) atomicAdd(bad,1);
            atomicAdd(seen+list[q],1);
        }
        __syncwarp(0xffffffffu);
    }
}
int main() {
    // Empty, tails around warp boundaries, uniform, heavy-only, mixed and large.
    const int cases[][8]={{0,0,0,0,0,0,0,0},{1,0,31,32,33,63,64,65},
        {4097,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,8193},
        {100003,2,35,90,0,7,512,13}};
    for(const auto& sizes:cases) {
        int offsets[9]={0};
        for(int b=0;b<8;b++) offsets[b+1]=offsets[b]+sizes[b];
        int n=offsets[8];
        // Permutation into odd cell indices models an excluded/masked subset.
        std::vector<int> list(n),seen(2*n+1);
        for(int q=0;q<n;q++) list[q]=2*(n-1-q)+1;
        int *d_offsets,*heads,*d_list,*d_seen,*bad;
        CHECK(cudaMalloc(&d_offsets,sizeof(offsets))); CHECK(cudaMalloc(&heads,8*sizeof(int)));
        CHECK(cudaMalloc(&d_list,(n+1)*sizeof(int))); CHECK(cudaMalloc(&d_seen,seen.size()*sizeof(int)));
        CHECK(cudaMalloc(&bad,sizeof(int)));
        CHECK(cudaMemcpy(d_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice));
        if(n) CHECK(cudaMemcpy(d_list,list.data(),n*sizeof(int),cudaMemcpyHostToDevice));
        for(int blocks: {1,3,80}) for(int threads: {32,128,256}) for(int repeat=0;repeat<2;repeat++) {
            CHECK(cudaMemset(heads,0,8*sizeof(int))); CHECK(cudaMemset(d_seen,0,seen.size()*sizeof(int)));
            CHECK(cudaMemset(bad,0,sizeof(int)));
            visit<<<blocks,threads>>>(d_offsets,heads,d_list,d_seen,bad);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(seen.data(),d_seen,seen.size()*sizeof(int),cudaMemcpyDeviceToHost));
            int errors=0; CHECK(cudaMemcpy(&errors,bad,sizeof(int),cudaMemcpyDeviceToHost));
            for(int i=0;i<(int)seen.size();i++) if(seen[i]!=(i%2)) errors++;
            if(errors) { fprintf(stderr,"FAIL n=%d blocks=%d threads=%d errors=%d\n",n,blocks,threads,errors); return 1; }
        }
        CHECK(cudaFree(d_offsets)); CHECK(cudaFree(heads)); CHECK(cudaFree(d_list));
        CHECK(cudaFree(d_seen)); CHECK(cudaFree(bad));
    }
    puts("PASS: queue exact-once, masked cells, bucket tails, empty queues, reset and worker counts");
}
