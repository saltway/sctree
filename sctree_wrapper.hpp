#ifndef __FPTREE_WRAPPER_HPP__
#define __FPTREE_WRAPPER_HPP__

#include "tree_api.hpp"
#include "sctree.h"

#include <cstring>
#include <libpmemobj.h>

// #define DEBUG_MSG

class ztree_wrapper : public tree_api
{
public:
    ztree_wrapper(const char *nvm_addr);
    virtual ~ztree_wrapper();

    virtual bool find(const char *key, size_t key_sz, char *value_out) override;
    virtual bool insert(const char *key, size_t key_sz, const char *value, size_t value_sz) override;
    virtual bool update(const char *key, size_t key_sz, const char *value, size_t value_sz) override;
    virtual bool remove(const char *key, size_t key_sz) override;
    virtual int scan(const char *key, size_t key_sz, int scan_sz, char *&values_out) override;

private:
    btree *ztree;
    FILE *fp = NULL;
    uint64_t count = 0;
};

thread_local char k[128];

ztree_wrapper::ztree_wrapper(const char *nvm_addr)
{
    leaf_class.alignment = 256;
    leaf_class.header_type = POBJ_HEADER_NONE;
    leaf_class.unit_size = LEAF_PAGESIZE;
    leaf_class.units_per_block = 1000;

    inner_class.alignment = 256;
    inner_class.header_type = POBJ_HEADER_NONE;
    inner_class.unit_size = INNER_PAGESIZE;
    inner_class.units_per_block = 1000;
    TOID(btree)
    bt = TOID_NULL(btree);
    PMEMobjpool *pop;

    pop = pmemobj_create(nvm_addr, "btree", 10000000000,
                         0666); // make 1GB memory pool
    bt = POBJ_ROOT(pop, btree);
    pmemobj_ctl_set(pop, "heap.alloc_class.new.desc", &leaf_class);
    pmemobj_ctl_set(pop, "heap.alloc_class.new.desc", &inner_class);
    D_RW(bt)->constructor(pop);
    ztree = (btree *)D_RW(bt);
    fp=fopen("sample_i.txt","a");
    puts("tree initialized");
    // ztree->btree_insert(123, (char *)123);
    // ztree->btree_insert(234, (char *)234);
    // // ztree->btree_delete(234);
    // char* ret = ztree->btree_search(234);
    // exit(0);
}

ztree_wrapper::~ztree_wrapper()
{
    fclose(fp);
    fp = NULL;
}

bool ztree_wrapper::find(const char *key, size_t key_sz, char *value_out)
{
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    char *ret = ztree->btree_search(k);
    if (ret)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool ztree_wrapper::insert(const char *key, size_t key_sz, const char *value, size_t value_sz)
{

    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    uint64_t v = *reinterpret_cast<uint64_t *>(const_cast<char *>(value));
    // fprintf(fp,"%lu\n", k);
    // printf("key is %ul\n",k);
    ztree->btree_insert(k, (char *)v);
    // count++;
    return true;
}

bool ztree_wrapper::update(const char *key, size_t key_sz, const char *value, size_t value_sz)
{
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    uint64_t v = *reinterpret_cast<uint64_t *>(const_cast<char *>(value));

    ztree->btree_update(k, (char *)v);
    return true;
}

bool ztree_wrapper::remove(const char *key, size_t key_sz)
{
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    ztree->btree_delete(k);
    return true;
}

int ztree_wrapper::scan(const char *key, size_t key_sz, int scan_sz, char *&values_out)
{
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    ztree->btree_search_range(k, LONG_MAX, (unsigned long *)values_out, scan_sz);
    return scan_sz;
}
#endif