libsctree.so is a library generated by pibench and can be directly implemented on pibench (https://github.com/sfu-dis/pibench-ep2). We also provide the wrapper files of pibench, sctree_wrapper.hpp and sctree_wrapper.cpp. Time mode is recommended as operation mode will generate duplicated payloads (load and operation phase start with the same position).

sctree.h is the source code of sc-tree

sctree.cpp is our test code on sc-tree and workload path is specified in this code.

fastfair.h is our complement version of FAST&FAIR(https://github.com/DICL/FAST_FAIR) in our experiments. We fix some bugs and complement the update function. To run this code on pibench, the first payload should be skipped as (0,0) is not supported in FAST&FAIR

generator is an executable file that generates simple random workloads for tests

to perform simple test on sc-tree, PMDK and openmp (if you use our test code) should be installed first. Then make sure rtm is available on your system (https://www.kernel.org/doc/html/latest/admin-guide/hw-vuln/tsx_async_abort.html#taa-mitigation-control-command-line). 

sc-tree provides the same interfaces as FAST&FAIR

the major compiling flags include -lpmemobj -mrtm 

-fopenmp is needed in compiling sctree.cpp

for example, run the commands as following

./generator (need several minutes)

g++  sctree.cpp -o sctree -std=c++11 -lpmemobj -mrtm -fopenmp -O3

numactl --cpubind=0 --membind=0 ./sctree -n $operation_number -t $thread_number -p $pm_path -o 1 (1 means warmup + test, 0 means only warmup, 2 means direct operation without warmup)
