U_CFLAGS=$(CFLAGS) -O3 --std=c++2a -fopenmp

targets=$(basename $(wildcard **.cpp))

all: $(targets)

clean:
	rm *.out

$(targets):
	$(CXX) $@.cpp -o$@.out $(U_CFLAGS) 