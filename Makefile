PLATFORM ?= CUDA

3RD_PARTY_INCLUDE := -I3rd-party/fmt/include/

TESTCASE ?= gemm
TEST_FILES := $(shell find tests -name "test_*.cpp" | sed 's@.*/\([^/]*\)\.cpp@\1@')
LINK_SO = $$(find build/bin/ -name 'libtest_*_$(plat).so')
TEST_BIN_FILES := $(wildcard build/test_*)
TEST_EXAMPLE ?= check_$(TESTCASE)_$(plat)

ifeq ($(and $(TESTCASE), $(PLATFORM)), gemm CUDA)
	COMPILE_OPTIONS += -I3rd-party/cutlass/include
endif

ifeq ($(PLATFORM), CUDA)
	plat := cuda
	CXX := nvcc
	COMPILE_OPTIONS += -lcudart
else ifeq ($(PLATFORM), BANG)
	plat := bang
	CXX := cncc
	COMPILE_OPTIONS += -L/usr/local/neuware/lib64 -lcnrt -I/usr/local/neuware/include
endif

COMPILE_OPTIONS += -Ibuild/bin/ -lc -lm -Wl,-rpath=build/bin/ -lstdc++ 


.PHONY: build tests clean test format

build:
	@mkdir -p build/code
	@mkdir -p build/bin
	@cd build && cmake ..
	@make -C build $(TEST_FILES)

test: build
	@./build/test_$(TESTCASE)
	@cp tests/$(TEST_EXAMPLE).cpp build/bin/
	@gcc build/bin/$(TEST_EXAMPLE).cpp -o build/bin/$(TEST_EXAMPLE) $(COMPILE_OPTIONS) $(LINK_SO)
	@./build/bin/$(TEST_EXAMPLE)

tests: build
	@$(foreach file, $(TEST_BIN_FILES), ./$(file);)

format:
	@./tools/format ./include
	@./tools/format ./source
	@./tools/format ./test

clean:
	@rm -rf build/
