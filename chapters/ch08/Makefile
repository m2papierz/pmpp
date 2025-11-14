# Makefile for CUDA diffusion (Allen–Cahn) simulation Python interface

NVCC        = nvcc

# Single-arch, overridable (call: make CUDA_ARCH=sm_86)
CUDA_ARCH  ?= sm_89

# Shared library for Python (inside ./build)
SHARED_LIB  = build/libdiffusion_cuda.so

# Compiler / linker flags
NVCCFLAGS   = -arch=$(CUDA_ARCH) -O3 -std=c++20 -Xcompiler -Wall
SHAREDFLAGS = -shared -Xcompiler -fPIC
LDFLAGS     = -lcudart

# Sources / objects
OBJDIR      = build
SOURCES     = diffusion.cu
OBJECTS     = $(OBJDIR)/diffusion.o

# Default target: build the shared library only
all: $(SHARED_LIB)

# Shared library for Python
$(SHARED_LIB): $(OBJECTS) | build
	@echo "Creating shared library $(SHARED_LIB)..."
	$(NVCC) $(NVCCFLAGS) $(SHAREDFLAGS) $(OBJECTS) -o $(SHARED_LIB) $(LDFLAGS)

# Compile CUDA source files with PIC
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c $< -o $@

# Clean target
clean:
	@echo "Cleaning..."
	rm -f $(OBJECTS) $(SHARED_LIB)

# Explicit target name
shared: $(SHARED_LIB)

# Debug / profile variants
debug: NVCCFLAGS += -g -G -lineinfo
debug: clean all

profile: NVCCFLAGS += -lineinfo
profile: clean all

.PHONY: all clean shared debug profile
