// #ifndef DIS
// #include "ztree_bitmap_lockbit.h"
// #else
// #include "ztree_bitmap_lockbit_distributed.h"
// #endif
// #define DRAM
#ifdef DRAM
#include "ztree_bitmap_lockbit_distributed_DRAM.h"
#else
#include "sctree.h"
#endif
#include <omp.h>

/*
 *  *file_exists -- checks if file exists
 *   */
static inline int file_exists(char const *file) { return access(file, F_OK); }

void clear_cache()
{
    // Remove cache
    int size = 256 * 1024 * 1024;
    char *garbage = new char[size];
    for (int i = 0; i < size; ++i)
        garbage[i] = i;
    for (int i = 100; i < size; ++i)
        garbage[i] += garbage[i - 100];
    delete[] garbage;
}

// MAIN
int main(int argc, char **argv)
{
    // Parsing arguments
    int numData = 0;
    int n_threads = 1;
    char *input_path = (char *)std::string("../sample_input.txt").data();
    char *persistent_path;
    int option = 0;

    srand(time(NULL));
    int c;
    while ((c = getopt(argc, argv, "n:w:t:i:p:o:")) != -1)
    {
        switch (c)
        {
        case 'n':
            numData = atoi(optarg);
            break;
        case 't':
            n_threads = atoi(optarg);
            break;
        case 'i':
            input_path = optarg;
        case 'p':
            persistent_path = optarg;
        case 'o':
            option = atoi(optarg);
        default:
            break;
        }
    }
    leaf_class.alignment = 256;
    leaf_class.header_type = POBJ_HEADER_NONE;
    leaf_class.unit_size = LEAF_PAGESIZE;
    leaf_class.units_per_block = 1000;

    inner_class.alignment = 256;
    inner_class.header_type = POBJ_HEADER_NONE;
    inner_class.unit_size = INNER_PAGESIZE;
    inner_class.units_per_block = 1000;

    long long elapsedTime;
    struct timespec start, end, tmp;
    btree *bt;
// Make or Read persistent pool
#ifdef DRAM
    bt = new btree();
#else
    TOID(btree)
    tree = TOID_NULL(btree);
    PMEMobjpool *pop;

    if (file_exists(persistent_path) != 0)
    {

        pop = pmemobj_create(persistent_path, "btree", 16000000000,
                             0666); // make 1GB memory pool
        tree = POBJ_ROOT(pop, btree);
        bt = D_RW(tree);
        pmemobj_ctl_set(pop, "heap.alloc_class.new.desc", &leaf_class);
        pmemobj_ctl_set(pop, "heap.alloc_class.new.desc", &inner_class);
        // pmemobj_ctl_get(pop, "heap.alloc_class.new.desc", &inner_class);
        // printf("%d\n", inner_class.alignment);
        // printf("%d\n", inner_class.unit_size);
        bt->constructor(pop);
    }
    else
    {
        pop = pmemobj_open(persistent_path, "btree");
        tree = POBJ_ROOT(pop, btree);
        bt = D_RW(tree);
        pmemobj_ctl_set(pop, "heap.alloc_class.new.desc", &leaf_class);
        pmemobj_ctl_set(pop, "heap.alloc_class.new.desc", &inner_class);
        cout << "data exists and begin recovering" << endl;
        cout << "current version : " << bt->version << endl;
        clock_gettime(CLOCK_MONOTONIC, &start);
        bt->recover(pop);
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsedTime =
            (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
        cout << "recovered in  " << elapsedTime / 1000 << " us" << endl;
    }

#endif

    // Reading data
    entry_key_t *keys = new entry_key_t[numData];

    ifstream ifs;
    ifs.open("sample_input.txt");

    if (!ifs)
    {
        cout << "input loading error!" << endl;
    }

    for (int i = 0; i < numData; ++i)
    {
        ifs >> keys[i];
    }
    ifs.close();

    long half_num_data = numData / 2;

    // Warm-up! Insert half of input size
    if (option < 2)
    {
#pragma omp parallel num_threads(n_threads)
        {
#pragma omp for schedule(static)
            for (int i = 0; i < half_num_data; ++i)
            {
                bt->btree_insert(keys[i], (char *)keys[i]);
            }
        }
        cout << "xl Warm-up!" << endl;

        bt->randScounter();
        if (option == 0)
            exit(0);
    }

    clear_cache();

#ifndef MIXED
     Search
     clock_gettime(CLOCK_MONOTONIC, &start);
 #pragma omp parallel num_threads(n_threads)
     {
 #pragma omp for schedule(static)
         for (int i = 0; i < half_num_data; ++i)
         {
             bt->btree_search(keys[i]);
         }
     }
     clock_gettime(CLOCK_MONOTONIC, &end);
     elapsedTime =
         (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
     cout << "Concurrent searching with " << n_threads
          << " threads (Kops) : " << (half_num_data * 1000000) / elapsedTime << endl;

     clear_cache();

    // Insert
    // write_cnt = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);

#pragma omp parallel num_threads(n_threads)
    {
#pragma omp for schedule(static)
        for (int i = half_num_data; i < numData; ++i)
        {
            bt->btree_insert(keys[i], (char *)keys[i]);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsedTime =
        (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    cout << "Concurrent inserting with " << n_threads
         << " threads (Kops) : " << (half_num_data * 1000000) / elapsedTime << endl;
    // fprintf(fp, "    insertion: %d,     %lld\n", n_threads, elapsedTime / (half_num_data));
    printf("the tree height is %d\n", bt->height);

    // printf("additional writes is %d\n", write_cnt);
    //   // Search
    // clock_gettime(CLOCK_MONOTONIC, &start);

    // for (int tid = 0; tid < n_threads; tid++)
    // {
    //   int from = half_num_data + data_per_thread * tid;
    //   int to = (tid == n_threads - 1) ? numData : from + data_per_thread;
    //   //     from += half_num_data;
    //   // to += half_num_data;
    //   auto f = async(
    //       launch::async,
    //       [&bt, &keys](int from, int to)
    //       {
    //         for (int i = from; i < to; ++i){
    //          D_RW(bt)->btree_search(keys[i]);
    //         }
    //       },
    //       from, to);
    //   futures.push_back(move(f));
    // }
    // for (auto &&f : futures)
    //   if (f.valid())
    //     f.get();

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // elapsedTime =
    //     (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    // cout << "Concurrent searching with " << n_threads
    //      << " threads (Kops) : " << (half_num_data*1000000)/elapsedTime << endl;
    // for(int i = 0; i< numData;i++){
    //   D_RW(bt)->btree_search(keys[i]);
    // }
    // puts("end test");
#else
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int tid = 0; tid < n_threads; tid++)
    {
        int from = half_num_data + data_per_thread * tid;
        int to = (tid == n_threads - 1) ? numData : from + data_per_thread;

        auto f = async(
            launch::async,
            [&bt, &keys, &half_num_data](int from, int to)
            {
                for (int i = from; i < to; ++i)
                {
                    int sidx = i - half_num_data;

                    int jid = i % 4;
                    switch (jid)
                    {
                    case 0:
                        D_RW(bt)->btree_insert(keys[i], (char *)keys[i]);
                        for (int j = 0; j < 4; j++)
                            D_RW(bt)->btree_search(
                                keys[(sidx + j + jid * 8) % half_num_data]);
                        // D_RW(bt)->btree_delete(keys[i]);
                        break;

                    case 1:
                        for (int j = 0; j < 3; j++)
                            D_RW(bt)->btree_search(
                                keys[(sidx + j + jid * 8) % half_num_data]);
                        D_RW(bt)->btree_insert(keys[i], (char *)keys[i]);
                        D_RW(bt)->btree_search(
                            keys[(sidx + 3 + jid * 8) % half_num_data]);
                        break;
                    case 2:
                        for (int j = 0; j < 2; j++)
                            D_RW(bt)->btree_search(
                                keys[(sidx + j + jid * 8) % half_num_data]);
                        D_RW(bt)->btree_insert(keys[i], (char *)keys[i]);
                        for (int j = 2; j < 4; j++)
                            D_RW(bt)->btree_search(
                                keys[(sidx + j + jid * 8) % half_num_data]);
                        break;
                    case 3:
                        for (int j = 0; j < 4; j++)
                            D_RW(bt)->btree_search(
                                keys[(sidx + j + jid * 8) % half_num_data]);
                        D_RW(bt)->btree_insert(keys[i], (char *)keys[i]);
                        break;
                    default:
                        break;
                    }
                }
            },
            from, to);
        futures.push_back(move(f));
    }

    for (auto &&f : futures)
        if (f.valid())
            f.get();

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsedTime =
        (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
    cout << "Concurrent inserting and searching with " << n_threads
         << " threads (Kops) : " << (half_num_data * 5000000) / elapsedTime << endl;
#endif

    delete[] keys;
#ifndef DRAM
    pmemobj_close(pop);
#endif
    return 0;
}
