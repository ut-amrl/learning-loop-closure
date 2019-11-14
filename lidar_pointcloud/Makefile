C_compiler=/usr/bin/gcc
CXX_compiler=/usr/bin/g++

# acceptable build_types: Release/Debug/Profile
# build_type=Release
build_type=Release

all: build/CMakeLists.txt.copy
	$(info Build_type is [${build_type}])
	$(MAKE) --no-print-directory -C build

clean:
	rm -rf bin lib build msg_gen

build/CMakeLists.txt.copy: build CMakeLists.txt Makefile
	cd build && cmake -DCMAKE_BUILD_TYPE=$(build_type) \
		-DCMAKE_CXX_COMPILER=$(CXX_compiler) \
		-DCMAKE_C_COMPILER=$(C_compiler) ..
	cp CMakeLists.txt build/CMakeLists.txt.copy

build:
	mkdir -p build
