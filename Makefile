CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra -pthread
INC      := -Iinclude
BUILD    ?= build

all: $(BUILD)/bench $(BUILD)/test_hnsw

$(BUILD)/%: src/%.cpp include/hnsw.h include/dataset.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@
$(BUILD)/test_hnsw: tests/test_hnsw.cpp include/hnsw.h include/dataset.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/test_hnsw
	$(BUILD)/test_hnsw

# sanitizer builds of the test suite (-O1 for usable stack traces)
tsan:
	mkdir -p $(BUILD) && $(CXX) -std=c++17 -O1 -g -fsanitize=thread -pthread $(INC) tests/test_hnsw.cpp -o $(BUILD)/test_tsan && $(BUILD)/test_tsan
asan:
	mkdir -p $(BUILD) && $(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -pthread $(INC) tests/test_hnsw.cpp -o $(BUILD)/test_asan && $(BUILD)/test_asan

clean:
	rm -rf $(BUILD)

.PHONY: all test tsan asan clean
