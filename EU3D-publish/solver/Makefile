# CPU/MPI framework and CUDA solver. Target: CX3 L40S, CUDA 12.6.
# Compile C++ with mpicxx and CUDA with nvcc using the MPI host compiler.

MPICXX   ?= mpicxx
NVCC     ?= nvcc
ARCH     ?= -arch=sm_89
NUMERIC_MODE ?= precise

# precise: disable contraction; native: use default floating-point operations.
ifeq ($(NUMERIC_MODE),precise)
CPU_FPFLAGS  = -ffp-contract=off
CUDA_FPFLAGS = --fmad=false --prec-div=true --prec-sqrt=true
else ifeq ($(NUMERIC_MODE),native)
CPU_FPFLAGS  =
CUDA_FPFLAGS =
else
$(error NUMERIC_MODE must be precise or native)
endif

COMMON_CPUFLAGS = -std=c++17 -O3 $(CPU_FPFLAGS) -I./include
CPUFLAGS  = $(COMMON_CPUFLAGS) -DUSE_GPU
BASEFLAGS = $(COMMON_CPUFLAGS)
NVCCFLAGS = -O3 -std=c++14 $(ARCH) $(CUDA_FPFLAGS) -I./include -I./src --expt-relaxed-constexpr \
            -ccbin $(MPICXX)

CPU_SRC = $(wildcard cpu_src/*.cpp)
CU_SRC  = src/gpu_solver.cu src/boundary.cu src/advection.cu src/reaction.cu src/comm.cu

# CPU target excludes the GPU driver.
BASE_SRC = $(filter-out cpu_src/Euler_GPU.cpp,$(CPU_SRC))

CPU_OBJ = $(patsubst cpu_src/%.cpp,obj/%.o,$(CPU_SRC))
CU_OBJ  = $(patsubst src/%.cu,obj/%.o,$(CU_SRC))
BASE_OBJ = $(patsubst cpu_src/%.cpp,obj_cpu/%.o,$(BASE_SRC))

TARGET  = EU3D_GPU
CPU_TARGET = EU3D_CPU

.PHONY: all cpu clean test_kernel test_dlb_schedule

all: $(TARGET)

cpu: $(CPU_TARGET)

# CPU framework with the GPU entry point enabled.
obj/%.o: cpu_src/%.cpp
	@mkdir -p obj
	$(MPICXX) $(CPUFLAGS) -c $< -o $@

# CUDA kernels with MPI headers from mpicxx.
obj/%.o: src/%.cu
	@mkdir -p obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

obj_cpu/%.o: cpu_src/%.cpp
	@mkdir -p obj_cpu
	$(MPICXX) $(BASEFLAGS) -c $< -o $@

# Link CUDA and MPI objects.
$(TARGET): $(CPU_OBJ) $(CU_OBJ)
	$(NVCC) $(ARCH) -ccbin $(MPICXX) $^ -o $@
	@echo "Build OK: $(TARGET)"

$(CPU_TARGET): $(BASE_OBJ)
	$(MPICXX) $^ -o $@
	@echo "Build OK: $(CPU_TARGET)"

# Standalone advection smoke test.
test_kernel: src/advection.cu src/test_advection.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_dlb_schedule: src/test_dlb_schedule.cpp src/dlb_schedule.hpp
	$(MPICXX) -std=c++17 -O2 -I./src src/test_dlb_schedule.cpp -o $@

clean:
	rm -rf obj obj_cpu $(TARGET) $(CPU_TARGET) test_kernel test_dlb_schedule

obj/reaction.o: src/warp_queue.cuh

test_warp_queue: src/test_warp_queue.cu src/warp_queue.cuh
	$(NVCC) $(NVCCFLAGS) src/test_warp_queue.cu -o $@
