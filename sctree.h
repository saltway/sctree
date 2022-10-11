/*
   inner FAST&FAIR and leaf with bitmap, distributed header, read commmitted
*/

#include <cassert>
#include <climits>
#include <fstream>
#include <future>
#include <iostream>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <libpmemobj.h>
#include <libpmem.h>
#include <immintrin.h>

#define INNER_PAGESIZE 512
#define LEAF_PAGESIZE 256
#define CPU_FREQ_MHZ (1994)
#define DELAY_IN_NS (1000)
#define CACHE_LINE_SIZE 64
#define QUERY_NUM 25
#define IS_FORWARD(c) (c % 2 == 0)

#define prefetcht0(mem_var)            \
  __asm__ __volatile__("prefetcht0 %0" \
                       :               \
                       : "m"(mem_var))
#define setbitmap0(x, y) asm("btr %2, %0" \
                             : "=m"(x)    \
                             : "m"(x), "r"(y))
//#define setbitmap0(x,y)(x &= (~(0x01L<<(y))))
#define setbitmap1(x, y) asm("bts %2, %0" \
                             : "=m"(x)    \
                             : "m"(x), "r"(y))
#define complement_bit(x, y) asm("btc %2, %0" \
                                 : "=m"(x)    \
                                 : "m"(x), "r"(y))
//#define setbitmap1(x,y) (x |= 0x01L<<(y))
#define getbitmap(x, y) ((x & (0x01L << (y))) >> (y))
#define get0(x, pos) \
  uint64_t x_r = ~x; \
  asm("bsf %1, %0"   \
      : "=r"(pos)    \
      : "r"(x_r))
#define get1(x, pos) \
  asm("bsf %1, %0"   \
      : "=r"(pos)    \
      : "r"(x))
#define llock(x, y, z)   \
  asm("clc");            \
  asm("lock bts %1, %0"  \
      :                  \
      : "m"(x), "r"(z)); \
  asm("adc %2, %1"       \
      : "=m"(y)          \
      : "m"(y), "r"(1))
class btree;
class leaf_page;
class inner_page;

POBJ_LAYOUT_BEGIN(mytree);
POBJ_LAYOUT_ROOT(mytree, btree);
POBJ_LAYOUT_TOID(mytree, PMEMrwlock);
POBJ_LAYOUT_TOID(mytree, leaf_page);
POBJ_LAYOUT_TOID(mytree, inner_page);
POBJ_LAYOUT_END(mytree);

using entry_key_t = uint64_t;
int clflush_cnt = 0;
unsigned char BitsSetTable256[256] = {0};
double inner_search_elapsed_time = 0;
int inner_search_cnt = 0;
double flush_elapsed_time = 0;
int range_node_cnt = 1;
int write_cnt = 1;
int clwb_cnt = 0;
long node_cnt = 0;
int range_cnt;
// int node_cnt = 0;
struct pobj_alloc_class_desc leaf_class;
struct pobj_alloc_class_desc inner_class;

static inline void cpu_pause() { __asm__ volatile("pause" ::
                                                      : "memory"); }

static inline unsigned long read_tsc(void)
{
  unsigned long var;
  unsigned int hi, lo;

  asm volatile("rdtsc"
               : "=a"(lo), "=d"(hi));
  var = ((unsigned long long int)hi << 32) | lo;

  return var;
}
static inline void persist_page(void *data, int len)
{
  // asm volatile("sfence" ::: "memory");

  // pmemobj_persist(pop, data, len);
  uintptr_t uptr;
  for (uptr = (uintptr_t)data & ~(63);
       uptr < (uintptr_t)data + len; uptr += 64)
    asm volatile(".byte 0x66; xsaveopt %0"
                 : "+m"(*(volatile char *)uptr));
  asm volatile("sfence" ::
                   : "memory");
}

static inline void persist(void *data, int len)
{
  // asm volatile("sfence" ::: "memory");
  //  asm volatile("clwb %0" : "+m"(*(volatile char *)data));
  //  pmemobj_flush(pop, data, len);
  //  pmem_flush(data, len);

  asm volatile(".byte 0x66; xsaveopt %0"
               : "+m"(*(volatile char *)data));
  asm volatile("sfence" ::
                   : "memory");
}
static inline void inner_persist_page(void *data, int len)
{
  // asm volatile("sfence" ::: "memory");
  // pmemobj_persist(pop, data, len);
  uintptr_t uptr;
  for (uptr = (uintptr_t)data & ~(63);
       uptr < (uintptr_t)data + len; uptr += 64)
    asm volatile(".byte 0x66; xsaveopt %0"
                 : "+m"(*(volatile char *)uptr));
  asm volatile("sfence" ::
                   : "memory");
}

static inline void inner_persist(void *data, int len)
{
  // asm volatile("sfence" ::: "memory");
  // asm volatile("clwb %0" : "+m"(*(volatile char *)data));
  // pmemobj_persist(pop, data, len);
  asm volatile(".byte 0x66; xsaveopt %0"
               : "+m"(*(volatile char *)data));
  asm volatile("sfence" ::
                   : "memory");
}

using namespace std;

class btree
{
public:
  int height;
  TOID(inner_page)
  root;
  PMEMobjpool *pop;
  uint64_t version;

public:
  btree();
  void constructor(PMEMobjpool *);
  void setNewRoot(TOID(inner_page));
  void btree_insert(entry_key_t, char *);
  void btree_insert_internal(char *, entry_key_t, char *, uint32_t);
  void btree_delete(entry_key_t);
  void btree_delete_internal(entry_key_t, char *, uint32_t, entry_key_t *,
                             bool *, inner_page **, uint64_t *);
  void btree_update(entry_key_t, char *);
  char *btree_search(entry_key_t);
  void btree_search_range(entry_key_t, entry_key_t, unsigned long *, int);
  void randScounter();
  void recover(PMEMobjpool *);
  friend class leaf_page;
  friend class inner_page;
};
struct meta0
{
  uint64_t bitmap : 14;
  uint64_t split : 1;
  uint64_t active : 1;
  uint64_t sibling_ptr : 48;
};
struct meta1
{
  uint64_t level : 15;
  uint64_t dummy : 1;
  uint64_t lock : 8;
  uint64_t timestamp : 40;
};
class leaf_header
{
private:
  uint64_t level : 15;
  uint64_t dummy : 1;
  uint64_t lock : 8;
  uint64_t timestamp : 40;

  uint64_t lowest; // 8 byte

#ifdef FP
  uint8_t fps[24]; // 8 byte
#endif
  friend class leaf_page;
  friend class btree;

public:
  void constructor(PMEMobjpool *pop)
  {
    level = 0;
    lowest = 0;
    lock = 0;
  }

  ~leaf_header()
  {
  }
};
class inner_header
{
private:
  uint16_t level : 15;
  uint16_t lock : 1;
  int16_t last_index;       // 2 bytes
  uint16_t switch_counter;  // 2 bytes
  uint16_t is_deleted;      // 2 bytes
  uint64_t lowest;          // 8 byte
  uint64_t sibling_ptr;     // 8 bytes
  inner_page *leftmost_ptr; // 8 bytes
  uint64_t timestamp : 48;  // 8 bytes
  uint64_t active : 16;
  uint64_t highest; // 8 bytes

  friend class inner_page;
  friend class btree;

public:
  void constructor(PMEMobjpool *pop)
  {
    // rwlock_v = new pthread_rwlock_t;
    // if (pthread_rwlock_init(rwlock_v, NULL))
    // {
    //   perror("lock init fail");
    //   exit(1);
    // }

    leftmost_ptr = NULL;
    sibling_ptr = 0;
    switch_counter = 0;
    last_index = -1;
    is_deleted = false;
    lowest = 0;
    lock = 0;
    highest = ULONG_MAX;
  }

  ~inner_header()
  {
  }
};
class entry
{
private:
  entry_key_t key; // 8 bytes
  char *ptr;       // 8 bytes
public:
  void constructor()
  {
    key = ULONG_MAX;
    ptr = NULL;
  }

  friend class leaf_page;
  friend class inner_page;
  friend class btree;
};

const int leaf_cardinality = (LEAF_PAGESIZE - sizeof(leaf_header) - sizeof(meta0)) / sizeof(entry);
// const int leaf_cardinality = 116;
const int inner_cardinality = (INNER_PAGESIZE - sizeof(inner_header)) / sizeof(entry);
const int count_in_line = CACHE_LINE_SIZE / sizeof(entry);

////////////////////////////////////////////////////////////////FAST&FAIR inner node
class inner_page
{
private:
  inner_header hdr;                 // header in persistent memory, 16 bytes
  entry records[inner_cardinality]; // slots in persistent memory, 16 bytes * n

public:
  friend class btree;

  void constructor(btree *bt, PMEMobjpool *pop, uint32_t level = 0)
  {

    hdr.constructor(pop);
    for (int i = 0; i < inner_cardinality; i++)
    {
      records[i].key = ULONG_MAX;
      records[i].ptr = NULL;
    }

    hdr.level = level;
    records[0].ptr = NULL;
    hdr.timestamp = bt->version;
    inner_persist_page(this, sizeof(inner_page));
    // += sizeof(inner_page);
  }

  // this is called when tree grows
  void constructor(PMEMobjpool *pop, inner_page *left, entry_key_t key, inner_page *right, uint32_t level = 0)
  {
    hdr.constructor(pop);
    for (int i = 0; i < inner_cardinality; i++)
    {
      records[i].key = ULONG_MAX;
      records[i].ptr = NULL;
    }
    hdr.leftmost_ptr = left;
    hdr.level = level;
    records[0].key = key;
    records[0].ptr = (char *)right;
    records[1].ptr = NULL;

    hdr.last_index = 0;
    inner_persist_page(this, sizeof(inner_page));
    // node_cnt += sizeof(inner_page);
  }

  inline int count()
  {
    uint8_t previous_switch_counter;
    int count = 0;
    do
    {
      previous_switch_counter = hdr.switch_counter;
      count = hdr.last_index + 1;

      while (count >= 0 && records[count].ptr != NULL)
      {
        if (IS_FORWARD(previous_switch_counter))
          ++count;
        else
          --count;
      }

      if (count < 0)
      {
        count = 0;
        while (records[count].ptr != NULL)
        {
          ++count;
        }
      }

    } while (previous_switch_counter != hdr.switch_counter);

    return count;
  }
  inline bool remove_key(PMEMobjpool *pop, entry_key_t key)
  {
    // Set the switch_counter
    if (IS_FORWARD(hdr.switch_counter))
      ++hdr.switch_counter;

    bool shift = false;
    int i;
    for (i = 0; records[i].ptr != NULL; ++i)
    {
      if (!shift && records[i].key == key)
      {
        records[i].ptr =
            (i == 0) ? (char *)hdr.leftmost_ptr : records[i - 1].ptr;
        shift = true;
      }

      if (shift)
      {
        records[i].key = records[i + 1].key;
        records[i].ptr = records[i + 1].ptr;

        // flush
        uint64_t records_ptr = (uint64_t)(&records[i]);
        int remainder = records_ptr % CACHE_LINE_SIZE;
        bool do_flush =
            (remainder == 0) ||
            ((((int)(remainder + sizeof(entry)) / CACHE_LINE_SIZE) == 1) &&
             ((remainder + sizeof(entry)) % CACHE_LINE_SIZE) != 0);
        if (do_flush)
        {
          pmemobj_persist(pop, (void *)records_ptr, CACHE_LINE_SIZE);
        }
      }
    }

    if (shift)
    {
      --hdr.last_index;
    }
    return shift;
  }

  bool remove(btree *bt, entry_key_t key, bool only_rebalance = false,
              bool with_lock = true)
  {
    if (hdr.timestamp != bt->version)
    {
      hdr.active = 1;
    }
  Retry:
    if (_xbegin() != _XBEGIN_STARTED)
    {
      goto Retry;
    }
    if (hdr.timestamp != bt->version)
    {
      hdr.timestamp = bt->version;
      hdr.lock = 0;
    }
    if (hdr.lock)
    {
      _xabort(1);
      goto Retry;
    }
    hdr.lock = 1;
    _xend();

    bool ret = remove_key(bt->pop, key);

    hdr.lock = 0;

    return ret;
  }
  inline void insert_key(PMEMobjpool *pop, entry_key_t key, char *ptr, int *num_entries,
                         bool flush = true, bool update_last_index = true)
  {
    // update switch_counter
    if (!IS_FORWARD(hdr.switch_counter))
      ++hdr.switch_counter;

    // FAST
    if (*num_entries == 0)
    { // this leaf_page is empty
      entry *new_entry = (entry *)&records[0];
      entry *array_end = (entry *)&records[1];
      new_entry->key = (entry_key_t)key;
      new_entry->ptr = (char *)ptr;

      array_end->ptr = (char *)NULL;

      if (flush)
      {
        inner_persist(this, CACHE_LINE_SIZE);
      }
    }
    else
    {
      int i = *num_entries - 1, inserted = 0, to_flush_cnt = 0;
      records[*num_entries + 1].ptr = records[*num_entries].ptr;
      if (flush)
      {
        if ((uint64_t) & (records[*num_entries + 1].ptr) % CACHE_LINE_SIZE == 0)
        {

          inner_persist(&records[*num_entries + 1].ptr, sizeof(char *));
        }
      }

      // FAST
      for (i = *num_entries - 1; i >= 0; i--)
      {
        if (key < records[i].key)
        {
          records[i + 1].ptr = records[i].ptr;
          records[i + 1].key = records[i].key;

          if (flush)
          {
            uint64_t records_ptr = (uint64_t)(&records[i + 1]);

            int remainder = records_ptr % CACHE_LINE_SIZE;
            bool do_flush =
                (remainder == 0) ||
                ((((int)(remainder + sizeof(entry)) / CACHE_LINE_SIZE) == 1) &&
                 ((remainder + sizeof(entry)) % CACHE_LINE_SIZE) != 0);
            if (do_flush)
            {
              inner_persist((void *)records_ptr, CACHE_LINE_SIZE);
              to_flush_cnt = 0;
            }
            else
              ++to_flush_cnt;
          }
        }
        else
        {
          records[i + 1].ptr = records[i].ptr;
          records[i + 1].key = key;
          records[i + 1].ptr = ptr;

          if (flush)
          {
            inner_persist(&records[i + 1], sizeof(entry));
          }
          inserted = 1;
          break;
        }
      }
      if (inserted == 0)
      {
        records[0].ptr = (char *)hdr.leftmost_ptr;
        records[0].key = key;
        records[0].ptr = ptr;
        if (flush)
          inner_persist(&records[0], sizeof(entry));
      }
    }

    if (update_last_index)
    {
      hdr.last_index = *num_entries;
    }
    ++(*num_entries);
  }

  // Insert a new key - FAST and FAIR
  inner_page *store(btree *bt, char *left, entry_key_t key, char *right, bool flush,
                    bool with_lock, inner_page *invalid_sibling = NULL)
  {

    if (with_lock)
    {
      if (hdr.timestamp != bt->version)
      {
        hdr.active = 1;
      }

    Retry0:
      // puts("leaf0");
      if (_xbegin() != _XBEGIN_STARTED)
      {
        goto Retry0;
      }
      if (hdr.timestamp != bt->version)
      {
        hdr.timestamp = bt->version;
        hdr.lock = 0;
      }
      _xend();
    Retry1:
      if (_xbegin() != _XBEGIN_STARTED)
      {
        goto Retry1;
      }
      if (hdr.lock)
      {
        _xabort(1);
        goto Retry1;
      }
      hdr.lock = 1;
      _xend();
    }
    if (hdr.is_deleted)
    {
      if (with_lock)
      {
        hdr.lock = 0;
      }

      return NULL;
    }
    // If this node has a sibling node,
    if ((hdr.sibling_ptr != 0) && ((inner_page *)hdr.sibling_ptr != invalid_sibling))
    {
      // Compare this key with the first key of the sibling
      TOID(inner_page)
      sib = pmemobj_oid(this);
      sib.oid.off = hdr.sibling_ptr;
      if (key > hdr.highest)

      {
        if (with_lock)
        {
          hdr.lock = 0;
        }
        return D_RW(sib)->store(bt, NULL, key, right, true, true, invalid_sibling);
      }
    }

    int num_entries = count();
    if (hdr.highest < records[num_entries - 1].key)
    {
      printf("num_entries is %d\n", num_entries);
      printf("highest is %lu\n", ULONG_MAX);
      printf("highest is %lu\n", hdr.highest);
      printf("highest key is %lu\n", records[num_entries - 1].key);
      puts("inner inconsistent");
      exit(0);
    }
    // FAST
    if (num_entries < inner_cardinality - 1)
    {
      insert_key(bt->pop, key, right, &num_entries, true);
      if (with_lock)
      {
        hdr.lock = 0;
      }
      return (inner_page *)pmemobj_oid(this).off;
    }
    else
    { // FAIR
      // overflow
      // create a new node
      TOID(inner_page)
      sibling;
      // POBJ_NEW(bt->pop, &sibling, inner_page, NULL, NULL);
      pmemobj_xalloc(bt->pop, (PMEMoid *)&sibling, sizeof(inner_page), 0, POBJ_CLASS_ID(inner_class.class_id), NULL, NULL);
      D_RW(sibling)->constructor(bt, bt->pop, hdr.level);
      inner_page *sibling_ptr = D_RW(sibling);
      int m = (int)ceil(num_entries / 2);
      entry_key_t split_key = records[m].key;

      // migrate half of keys into the sibling
      int sibling_cnt = 0;

      for (int i = m + 1; i < num_entries; ++i)
      {
        sibling_ptr->insert_key(bt->pop, records[i].key, records[i].ptr, &sibling_cnt,
                                false);
      }
      sibling_ptr->hdr.lowest = records[m].key;
      sibling_ptr->hdr.leftmost_ptr = (inner_page *)records[m].ptr;
      sibling_ptr->hdr.highest = hdr.highest;

      sibling_ptr->hdr.sibling_ptr = hdr.sibling_ptr;
      inner_persist_page(sibling_ptr, sizeof(inner_page));

      hdr.sibling_ptr = sibling.oid.off;
      hdr.highest = split_key;
      inner_persist(&hdr, sizeof(hdr));
      // set to NULL
      if (IS_FORWARD(hdr.switch_counter))
        hdr.switch_counter += 2;
      else
        ++hdr.switch_counter;
      records[m].ptr = NULL;
      inner_persist(&records[m], sizeof(entry));

      hdr.last_index = m - 1;
      inner_persist(&hdr.last_index, sizeof(int16_t));

      num_entries = hdr.last_index + 1;

      inner_page *ret;

      // insert the key
      if (key < split_key)
      {
        insert_key(bt->pop, key, right, &num_entries);
        ret = (inner_page *)pmemobj_oid(this).off;
      }
      else
      {
        sibling_ptr->insert_key(bt->pop, key, right, &sibling_cnt);
        ret = (inner_page *)sibling.oid.off;
      }

      // Set a new root or insert the split key to the parent
      if (D_RO(bt->root) == this)
      { // only one node can update the root ptr
        TOID(inner_page)
        new_root;
        // POBJ_NEW(bt->pop, &new_root, inner_page, NULL, NULL);
        pmemobj_xalloc(bt->pop, (PMEMoid *)&new_root, sizeof(inner_page), 0, POBJ_CLASS_ID(inner_class.class_id), NULL, NULL);
        D_RW(new_root)->constructor(bt->pop, (inner_page *)bt->root.oid.off,
                                    split_key, (inner_page *)sibling.oid.off,
                                    hdr.level + 1);
        bt->setNewRoot(new_root);
        if (with_lock)
        {
          hdr.lock = 0;
        }
      }
      else
      {
        if (with_lock)
        {
          hdr.lock = 0;
        }
        bt->btree_insert_internal(NULL, split_key, (char *)sibling.oid.off,
                                  hdr.level + 1);
      }

      return ret;
    }
  }

  char *linear_search(entry_key_t key)
  {
    int i = 1;
    uint8_t previous_switch_counter;
    char *ret = NULL;
    char *t;
    entry_key_t k;
  // internal node
  Again:
    do
    {
      previous_switch_counter = hdr.switch_counter;
      ret = NULL;

      if (IS_FORWARD(previous_switch_counter))
      {
        if (key < (k = records[0].key))
        {
          if ((t = (char *)hdr.leftmost_ptr) != records[0].ptr)
          {
            ret = t;
            continue;
          }
        }

        for (i = 1; records[i].ptr != NULL; ++i)
        {
          if (key < (k = records[i].key))
          {
            if ((t = records[i - 1].ptr) != records[i].ptr)
            {
              ret = t;
              break;
            }
          }
        }

        if (!ret)
        {
          ret = records[i - 1].ptr;
          continue;
        }
      }
      else
      {
        for (i = count() - 1; i >= 0; --i)
        {
          if (key >= (k = records[i].key))
          {
            if (i == 0)
            {
              if ((char *)hdr.leftmost_ptr != (t = records[i].ptr))
              {
                ret = t;
                break;
              }
            }
            else
            {
              if (records[i - 1].ptr != (t = records[i].ptr))
              {
                ret = t;
                break;
              }
            }
          }
        }
      }
    } while (hdr.switch_counter != previous_switch_counter);
    TOID(inner_page)
    temp;
    TOID_ASSIGN(temp, pmemobj_oid(this));
    temp.oid.off = hdr.sibling_ptr;

    if ((t = (char *)temp.oid.off) != NULL)
    {
      if (key >= hdr.highest)

        if (key >= D_RW(temp)->hdr.lowest)
        {
          return t;
        }
    }

    if (ret)
    {
      temp.oid.off = (uint64_t)ret;
      if (key < D_RW(temp)->hdr.lowest)

      {
        goto Again;
      }
      return ret;
    }
    else
    {
      return (char *)hdr.leftmost_ptr;
    }

    return NULL;
  }
};
/////////////////////////////////////////////////////////////////////////FAST&FAIR

/////////////////////////////////////////////////////////////////////////mytree
class leaf_page
{
private:
  leaf_header hdr;                 // header in persistent memory, 16 bytes
  entry records[leaf_cardinality]; // slots in persistent memory, 16 bytes * n
  uint64_t highest;
  struct meta0 meta0;

public:
  friend class btree;

  void constructor(btree *bt, pmemobjpool *pop, uint32_t level = 0)
  {
    meta0.sibling_ptr = 0;
    meta0.bitmap = 0;
    highest = ULONG_MAX;
    hdr.constructor(pop);
    hdr.level = level;
    hdr.timestamp = bt->version;
    pmemobj_persist(pop, &hdr, sizeof(hdr));
    // node_cnt += sizeof(leaf_page);
  }

  inline int count()
  {

    return __builtin_popcountl(meta0.bitmap);
  }
  inline void shift_key(entry_key_t key, char *ptr, int pos, bool backward = true)
  {
    if (!backward)
    {
      pos = (pos - (int)ceil(leaf_cardinality / 2)) * 2;
    }
    else if (pos == (int)ceil(leaf_cardinality / 2))
    {
      pos = 0;
    }

    records[pos].key = key;
    records[pos].ptr = ptr;
#ifdef FP
    hdr.fps[pos] = (uint8_t)key;
#endif
    setbitmap1(meta0, pos);
  }
  inline void insert_key(PMEMobjpool *pop, entry_key_t key, char *ptr,
                         bool flush = true, bool record = false)
  {
    uint64_t pos;
    get0(meta0.bitmap, pos);
    if (record)
    {
      hdr.lock = pos + 2;
    }

    // lock->padding[40] = pos + 1;
    records[pos].key = key;
    records[pos].ptr = ptr;
    // if (pos > leaf_cardinality-5){
    //   records[pos-4].ptr = ptr;
    //   // persist(&records[pos-4].ptr, CACHE_LINE_SIZE, pop);
    // }
#ifdef FP
    hdr.fps[pos] = (uint8_t)key;
    if (flush)
    {
      persist(&records[pos].ptr, CACHE_LINE_SIZE, pop);
    }

    // if (unlock)
    // {
    //   movnt64((uint64_t *)&hdr, m->word8B[0], false, true);
    //   hdr.lock = 0;
    // }
    else
    {
      setbitmap1(meta0, pos);
    }
    if (flush)
    {
      persist(&hdr, CACHE_LINE_SIZE, pop);
    }
#else
    // if (pos < (64 - sizeof(hdr)) / 16 && flush)
    // {
    //   setbitmap1(meta0, pos);
    //   // if (unlock)
    //   // {
    //   //   hdr.lock = 0;
    //   // }
    //   if (flush)
    //   {
    //     persist(&records[pos].ptr, CACHE_LINE_SIZE);
    //     // write_cnt++;
    //     // records[pos + 1].ptr = ptr + 1;
    //     // persist(&records[pos + 4].ptr, CACHE_LINE_SIZE, pop);
    //   }
    // }
    // else
    {
      if (flush)
      {
        persist(&records[pos].ptr, CACHE_LINE_SIZE);
      }
      setbitmap1(meta0, pos);
      // if (unlock)
      // {
      //   hdr.lock = 0;
      // }
      if (flush)
      {
        if (pos + 4 < leaf_cardinality)
        {
          // for (int i = pos + 4; i < leaf_cardinality; i = i + 4)
          // {
          //   records[i].ptr = records[i].ptr + 1;
          //   //  persist(&records[i].key, 1, pop);
          // }
          // records[pos].ptr = ptr + 1;
          // persist(&records[pos].key, 1, pop);
          // records[pos - 4].ptr = ptr + 1;
          // // write_cnt ++;
          // persist(&records[pos - 4].ptr, CACHE_LINE_SIZE, pop);
        }
        persist(&meta0, CACHE_LINE_SIZE);
        //  persist(&records[pos].ptr, CACHE_LINE_SIZE, pop);
      }
    }
#endif

    // lock->padding[40] = 0;
  }
  inline void permutation(uint8_t *slot_array, uint64_t bitmap, int sum, int offset)
  {
    int base = sum;
    for (int i = 0; i < offset; i++)
    {
      if (sum == 0)
      {
        slot_array[0] = i;
        sum++;
      }
      else
      {
        int inserted = 0;
        for (int j = sum - 1; j >= 0; j--)
        {
          if (records[i + base].key > records[slot_array[j]].key)
          {
            slot_array[j + 1] = i + base;
            inserted = 1;
            sum++;
            break;
          }
          else
          {
            slot_array[j + 1] = slot_array[j];
          }
        }
        if (inserted == 0)
        {
          slot_array[0] = i + base;
          sum++;
        }
      }
    }
  }
  // Insert a new key - FAST and FAIR
  leaf_page *store(btree *bt, char *left, entry_key_t key, char *right, bool flush,
                   leaf_page *invalid_sibling = NULL)
  {

    // If this node has a sibling node,
    if (hdr.timestamp != bt->version)
    {
      meta0.active = 1;
    }

  Retry0:
    // puts("leaf0");
    if (_xbegin() != _XBEGIN_STARTED)
    {
      goto Retry0;
    }
    if (hdr.timestamp != bt->version)
    {
      hdr.timestamp = bt->version;
      hdr.lock = 0;
    }
    _xend();
  Retry1:
    if (_xbegin() != _XBEGIN_STARTED)
    {
      goto Retry1;
    }
    if (hdr.lock)
    {
      _xabort(1);
      goto Retry1;
    }
    hdr.lock = 1;
    _xend();
    // persist(&hdr,1);

    uint64_t sib = meta0.sibling_ptr;
    TOID(leaf_page)
    sibling = pmemobj_oid(this);
    sibling.oid.off = sib;
    if (meta0.split)
    {
      if (sib != 0)
      {
        highest = D_RW(sibling)->hdr.lowest;
        if ((leaf_page *)D_RO(bt->root) == this)
        { // only one node can update the root ptr
          TOID(inner_page)
          new_root;
          // POBJ_NEW(bt->pop, &new_root, inner_page, NULL, NULL);
          pmemobj_xalloc(bt->pop, (PMEMoid *)&new_root, sizeof(inner_page), 0, POBJ_CLASS_ID(inner_class.class_id), NULL, NULL);
          D_RW(new_root)->constructor(bt->pop, (inner_page *)bt->root.oid.off,
                                      highest, (inner_page *)(sibling.oid.off),
                                      hdr.level + 1);
          bt->setNewRoot(new_root);
        }
        else
        {

          bt->btree_insert_internal(NULL, highest, (char *)(sibling.oid.off),
                                    hdr.level + 1);
        }
      }
      meta0.split = 0;
      persist(&meta0, CACHE_LINE_SIZE);
    }
    if (sib != 0)
    {
      if (key > highest)
      {
        hdr.lock = 0;
        persist(&hdr, CACHE_LINE_SIZE);
        return D_RW(sibling)->store(bt, NULL, key, right, true,
                                    invalid_sibling);
      }
    }
    int num_entries = count();

    // register int num_entries = hdr.last_index +1;
    //  FAST
    if (num_entries < leaf_cardinality)
    {
      insert_key(bt->pop, key, right, true, true);
      hdr.lock = 0;
      persist(&hdr, CACHE_LINE_SIZE);
      return this;
    }
    else
    { // FAIR
      // hdr.split0 = ~hdr.split0;

      // persist(&hdr, CACHE_LINE_SIZE, bt->pop);
      TOID(leaf_page)
      sibling;
      // POBJ_NEW(bt->pop, &sibling, leaf_page, NULL, NULL);
      pmemobj_xalloc(bt->pop, (PMEMoid *)&sibling, sizeof(leaf_page), 0, POBJ_CLASS_ID(leaf_class.class_id), NULL, NULL);
      D_RW(sibling)->constructor(bt, bt->pop, hdr.level);
      leaf_page *sibling_ptr = D_RW(sibling);
      uint8_t temp_slot_array[leaf_cardinality];
      permutation(temp_slot_array, meta0.bitmap, 0, leaf_cardinality);
      // register int m = (int)ceil(num_entries / 2);
      // m=58, cardinality=116
      int m = (int)ceil(num_entries / 2);
      entry_key_t split_key = records[temp_slot_array[m]].key;
      // printf(" the split_key is %d\n", split_key);
      //  migrate half of keys into the sibling

      uint64_t tempbitmap = meta0.bitmap;

      for (int i = m; i < num_entries; ++i)
      {
        setbitmap0(tempbitmap, (uint64_t)temp_slot_array[i]);
#ifdef SHIFT
        sibling_ptr->shift_key(records[temp_slot_array[i]].key, records[temp_slot_array[i]].ptr, i, true);
#else
        sibling_ptr->insert_key(bt->pop, records[temp_slot_array[i]].key, records[temp_slot_array[i]].ptr, NULL, false);
#endif
      }
      // printf("the split key and pos is :  %d  %d\n",split_key,m);

      sibling_ptr->meta0.sibling_ptr = meta0.sibling_ptr;
      sibling_ptr->hdr.lowest = split_key;
      sibling_ptr->highest = highest;
      persist_page(sibling_ptr, sizeof(leaf_page));

      struct meta0 temp_hdr;
      temp_hdr.sibling_ptr = sibling.oid.off;
      temp_hdr.bitmap = tempbitmap;
      temp_hdr.split = 1;

      meta0 = temp_hdr;
      highest = split_key;

      meta0.split = 0;
      persist(&meta0, sizeof(char *));
      // printf("pos of sib_p is %d\n", (uint64_t)(&hdr.sibling_ptr[1])%64);
      leaf_page *ret;
      // insert the key
      if (key < split_key)
      {
        insert_key(bt->pop, key, right, true, true);
        ret = (leaf_page *)pmemobj_oid(this).off;
      }
      else
      {
        sibling_ptr->insert_key(bt->pop, key, right, true, false);
        ret = (leaf_page *)sibling.oid.off;
      }

      // Set a new root or insert the split key to the parent
      if ((leaf_page *)D_RO(bt->root) == this)
      { // only one node can update the root ptr
        TOID(inner_page)
        new_root;
        // POBJ_NEW(bt->pop, &new_root, inner_page, NULL, NULL);
        pmemobj_xalloc(bt->pop, (PMEMoid *)&new_root, sizeof(inner_page), 0, POBJ_CLASS_ID(inner_class.class_id), NULL, NULL);
        D_RW(new_root)->constructor(bt->pop, (inner_page *)bt->root.oid.off,
                                    split_key, (inner_page *)(sibling.oid.off),
                                    hdr.level + 1);
        bt->setNewRoot(new_root);
        hdr.lock = 0;
        persist(&hdr, CACHE_LINE_SIZE);
      }
      else
      {
        hdr.lock = 0;
        persist(&hdr, CACHE_LINE_SIZE);
        bt->btree_insert_internal(NULL, split_key, (char *)(sibling.oid.off),
                                  hdr.level + 1);
      }
      return ret;
    }
  }

  char *linear_search(entry_key_t key, PMEMobjpool *pop)
  {

    // if (_xbegin() != _XBEGIN_STARTED)
    // {
    //   goto Retry;
    // }
    char *ret = NULL;
    char *t;
    entry_key_t k;
    ret = NULL;
    int pos = 0;
    // pthread_rwlock_t *lock_v = (pthread_rwlock_t *)(hdr.rwlock_v);
    // pthread_rwlock_rdlock(lock_v);
    // pmemobj_rwlock_rdlock(pop, lock_pointer);
    for (int i = 0; i < leaf_cardinality; ++i)
    {
//__builtin_prefetch((const void*)(&records + 8*i),0,0);
#ifdef FP
      if (getbitmap(hdr.bitmap, i) && hdr.fps[i] == (uint8_t)key && records[i].key == key)
      {
        ret = records[i].ptr;
        pos = i;
        break;
      }
#else
      if (getbitmap(meta0.bitmap, i) && records[i].key == key)
      {
        ret = records[i].ptr;
        pos = i;
        break;
      }
#endif
    }

    if (ret)
    {
      // _xend();
      if (hdr.lock != pos + 2)
      {
        return ret;
      }
      
    }
    pos = hdr.lock - 2 - leaf_cardinality;
    if (pos >= 0 && records[pos].key == key)
    {
      
      return records[pos].ptr;
    }
    TOID(leaf_page)
    sibling = pmemobj_oid(this);
    sibling.oid.off = meta0.sibling_ptr;
    if ((t = (char *)meta0.sibling_ptr) && key >= highest)
    {
      return t;
    }
    // pthread_rwlock_unlock(lock_v);
    // pmemobj_rwlock_unlock(pop, lock_pointer);
    // _xend();
    
    return NULL;
  }
  void remove(btree *bt, entry_key_t key)
  {
    if (hdr.timestamp != bt->version)
    {
      meta0.active = 1;
    }

  Retry0:
    // puts("leaf0");
    if (_xbegin() != _XBEGIN_STARTED)
    {
      goto Retry0;
    }
    if (hdr.timestamp != bt->version)
    {
      hdr.timestamp = bt->version;
      hdr.lock = 0;
    }
    _xend();
  Retry1:
    if (_xbegin() != _XBEGIN_STARTED)
    {
      goto Retry1;
    }
    if (hdr.lock)
    {
      _xabort(1);
      goto Retry1;
    }
    hdr.lock = 1;
    _xend();
    uint64_t sib = meta0.sibling_ptr;
    TOID(leaf_page)
    sibling = pmemobj_oid(this);
    sibling.oid.off = sib;
    if (sib != 0)
    {

      if (key > highest)
      {
        hdr.lock = 0;
        return D_RW(sibling)->remove(bt, key);
      }
    }
    char *ret = NULL;
    for (int i = 0; i < leaf_cardinality; ++i)
    {
      //__builtin_prefetch((const void*)(&records + 8*i),0,0);

      if (getbitmap(meta0.bitmap, i) && records[i].key == key)
      {
        ret = records[i].ptr;
        hdr.lock = i + 2 + leaf_cardinality;
        setbitmap0(meta0, i);
        persist(&meta0, 1);
        hdr.lock = 0;
        persist(&hdr, 1);
        return;
      }
    }
    if (!ret)
    {
      hdr.lock = 0;
      persist(&hdr, 1);
      // puts("key not exists");
      return;
    }
  }
  void update(btree *bt, entry_key_t key, char *ptr)
  {
    if (hdr.timestamp != bt->version)
    {
      meta0.active = 1;
    }

  Retry0:
    // puts("leaf0");
    if (_xbegin() != _XBEGIN_STARTED)
    {
      goto Retry0;
    }
    if (hdr.timestamp != bt->version)
    {
      hdr.timestamp = bt->version;
      hdr.lock = 0;
    }
    _xend();
  Retry1:
    if (_xbegin() != _XBEGIN_STARTED)
    {
      goto Retry1;
    }
    if (hdr.lock)
    {
      _xabort(1);
      goto Retry1;
    }
    hdr.lock = 1;
    _xend();

    uint64_t sib = meta0.sibling_ptr;
    TOID(leaf_page)
    sibling = pmemobj_oid(this);
    sibling.oid.off = sib;
    if (sib != 0)
    {

      if (key > highest)
      {
        hdr.lock = 0;
        persist(&hdr, 1);
        return D_RW(sibling)->update(bt, key, ptr);
      }
    }
    char *ret = NULL;
    for (int i = 0; i < leaf_cardinality; ++i)
    {
      //__builtin_prefetch((const void*)(&records + 8*i),0,0);

      if (getbitmap(meta0.bitmap, i) && records[i].key == key)
      {
        ret = records[i].ptr;
        records[i].ptr = ptr;
        hdr.lock = i + 2;
        persist(&records[i].ptr, 1);
        break;
      }
    }
    if (!ret)
    {
      puts("key not exists");
      exit(0);
    }
    hdr.lock = 0;
    persist(&hdr, 1);
  }
  void permutate(uint64_t *array, int last_index)
  {
    for (int i = 0; i < last_index; i++)
    {
      for (int j = 0; j < last_index; j++)
      {
        if (array[j] > array[j + 1])
        {
          array[j] += array[j + 1];
          array[j + 1] = array[j] - array[j + 1];
          array[j] = array[j] - array[j + 1];
        }
      }
    }
  }
  void linear_search_range(btree *bt, entry_key_t min, entry_key_t max,
                           unsigned long *buf, PMEMobjpool *pop, int number = 0)
  {
    int i, off = 0;
    int node_cnt = 1;
    uint64_t result[number];
    // struct timespec start, end;
    // double elapsed_time;
    leaf_page *current = this;
    // printf("the number is %d\n", number);
    while (current)
    {
      
     Retry:
       if (_xbegin() != _XBEGIN_STARTED)
       {
         goto Retry;
       }
      uint64_t temp_result[15];
      int temp_off = 0;
      entry_key_t tmp_key;
      entry_key_t max_key = 0;
      char *tmp_ptr;
      for (int i = 0; i < leaf_cardinality; ++i)
      {
        int pos = hdr.lock - 2 - leaf_cardinality;
        if (getbitmap(meta0.bitmap, i) || pos == i)
        {
          tmp_key = current->records[i].key;
          if (hdr.lock == i + 2)
          {
            continue;
          }

          if (tmp_key >= max_key)
          {
            max_key = tmp_key;
          }
          if (tmp_key > min && tmp_key < max)
          {
            temp_result[temp_off++] = current->records[i].key;
            // result[off++] = (uint64_t)(current->records[i].ptr);
            if (off + temp_off >= number)
            {
              _xend();
              permutate(temp_result, temp_off);
             memcpy(&result[off], temp_result, (temp_off) * 8);
              
              return;
            }
            // range_cnt++;
          }
        }
      }
        _xend();
      if (max_key >= max)
      {
        // hdr.lock = 0;
        // printf("the searched number is %d and the node accessed is %d\n", off, node_cnt);
        // _xend();
        permutate(temp_result, temp_off);
        memcpy(&result[off], temp_result, (temp_off) * 8);
        
        return;
      }
      permutate(temp_result, temp_off);
      memcpy(&result[off], temp_result, temp_off * 8);
      off += temp_off;
      if (current->meta0.sibling_ptr != 0)
      {
        TOID(leaf_page)
        sib;
        sib.oid = pmemobj_oid(this);
        sib.oid.off = current->meta0.sibling_ptr;
        current = D_RW(sib);
        range_node_cnt++;
        // hdr.lock = 0;
      }
      else
      {
        // puts("end scan with limited nodes");
        // hdr.lock = 0;
        // printf("the searched number is %d and the node accessed is %d\n", off, node_cnt);
        current = NULL;
      }
      
    }
    return;
  }
};

/*
 *  class btree
 */
void btree::constructor(PMEMobjpool *pool)
{
  pop = pool;
  TOID(leaf_page)
  new_root;
  pmemobj_xalloc(pool, (PMEMoid *)&new_root, sizeof(leaf_page), 0, POBJ_CLASS_ID(leaf_class.class_id), NULL, NULL);
  D_RW(new_root)->constructor(this, pool);
  root.oid = new_root.oid;
  persist(&root, sizeof(leaf_page));
  height = 1;
  version = 0;
  persist(this, sizeof(btree));

  // node_cnt += sizeof(btree);
}

void btree::setNewRoot(TOID(inner_page) new_root)
{
  root = new_root;
  inner_persist(&root, sizeof(TOID(inner_page)));
  ++height;
  pmemobj_persist(pop, &height, sizeof(height));
}

char *btree::btree_search(entry_key_t key)
{
  TOID(inner_page)
  p = root;

  while (D_RO(p)->hdr.level != 0)
  {
    p.oid.off = (uint64_t)D_RW(p)->linear_search(key);
  }
  TOID(leaf_page)
  ip;
  ip.oid = p.oid;
  uint64_t t;
  while ((t = (uint64_t)D_RW(ip)->linear_search(key, pop)) ==
         D_RO(ip)->meta0.sibling_ptr)
  {
    ip.oid.off = t;
    if (!t)
    {
      break;
    }
  }
  if (!t)
  {
    // printf("NOT FOUND %lu, t = %lu\n", key, t);
    return NULL;
  }

  return (char *)t;
}

// insert the key in the leaf node
void btree::btree_insert(entry_key_t key, char *right)
{ // need to be string
  // struct timespec start, end;

  TOID(inner_page)
  p = root;
  while (D_RO(p)->hdr.level != 0)
  {

    p.oid.off = (uint64_t)D_RW(p)->linear_search(key);
  }

  TOID(leaf_page)
  ip;
  ip.oid = p.oid;

  if (!D_RW(ip)->store(this, NULL, key, right, true))
  { // store
    btree_insert(key, right);
  }
}
void btree::btree_delete(entry_key_t key)
{
  TOID(inner_page)
  p = root;

  while (D_RO(p)->hdr.level != 0)
  {
    p.oid.off = (uint64_t)D_RW(p)->linear_search(key);
  }
  TOID(leaf_page)
  ip;
  ip.oid = p.oid;
  uint64_t t;
  D_RW(ip)->remove(this, key);
}
void btree::btree_search_range(entry_key_t min, entry_key_t max,
                               unsigned long *buf, int number = 0)
{

  TOID(inner_page)
  p = root;
  while (D_RO(p)->hdr.level != 0)
  {
    p.oid.off = (uint64_t)D_RW(p)->linear_search(min);
  }
  TOID(leaf_page)
  ip;
  ip.oid = p.oid;
  D_RW(ip)->linear_search_range(this, min, max, buf, pop, number);
}
void btree::btree_update(entry_key_t key, char *ptr)
{
  TOID(inner_page)
  p = root;

  while (D_RO(p)->hdr.level != 0)
  {
    p.oid.off = (uint64_t)D_RW(p)->linear_search(key);
  }
  TOID(leaf_page)
  ip;
  ip.oid = p.oid;
  uint64_t t;
  D_RW(ip)->update(this, key, ptr);
}
// store the key into the node at the given level
void btree::btree_insert_internal(char *left, entry_key_t key, char *right,
                                  uint32_t level)
{
  if (level > D_RO(root)->hdr.level)
    return;

  TOID(inner_page)
  p = root;

  while (D_RO(p)->hdr.level > level)
    p.oid.off = (uint64_t)D_RW(p)->linear_search(key);

  if (!D_RW(p)->store(this, NULL, key, right, true, true))
  {
    btree_insert_internal(left, key, right, level);
  } // else{
    //  clwb_cnt--;
    //  clwb_cnt--;
  //}
}
void btree::btree_delete_internal(entry_key_t key, char *ptr, uint32_t level,
                                  entry_key_t *deleted_key,
                                  bool *is_leftmost_node, inner_page **left_sibling, uint64_t *new_min = NULL)
{
  if (level > D_RO(root)->hdr.level)
    return;

  TOID(inner_page)
  p = root;

  while (D_RW(p)->hdr.level > level)
  {
    p.oid.off = (uint64_t)D_RW(p)->linear_search(key);
  }

  if ((char *)D_RO(p)->hdr.leftmost_ptr == ptr)
  {
    *is_leftmost_node = true;
    return;
  }

  *is_leftmost_node = false;

  for (int i = 0; D_RO(p)->records[i].ptr != NULL; ++i)
  {
    if (D_RO(p)->records[i].ptr == ptr)
    {
      if (i == 0)
      {
        if ((char *)D_RO(p)->hdr.leftmost_ptr != D_RO(p)->records[i].ptr)
        {
          *deleted_key = D_RO(p)->records[i].key;
          *new_min = D_RO(p)->records[i + 1].key;
          *left_sibling = D_RO(p)->hdr.leftmost_ptr;
          D_RW(p)->remove(this, *deleted_key, false, false);
          break;
        }
      }
      else
      {
        if (D_RO(p)->records[i - 1].ptr != D_RO(p)->records[i].ptr)
        {
          *deleted_key = D_RO(p)->records[i].key;
          *new_min = D_RO(p)->records[i + 1].key;
          *left_sibling = (inner_page *)D_RO(p)->records[i - 1].ptr;
          D_RW(p)->remove(this, *deleted_key, false, false);
          break;
        }
      }
    }
  }
}

void btree::randScounter()
{
  TOID(inner_page)
  leftmost = root;
  if (D_RO(leftmost)->hdr.level == 0)
  {
    return;
  }
  srand(time(NULL));
  if (root.oid.off)
  {
    do
    {
      TOID(inner_page)
      sibling = leftmost;
      while (sibling.oid.off)
      {
        D_RW(sibling)->hdr.switch_counter = rand() % 100;
        sibling.oid.off = D_RO(sibling)->hdr.sibling_ptr;
      }
      leftmost.oid.off = (uint64_t)D_RO(leftmost)->hdr.leftmost_ptr;
    } while (D_RO(leftmost)->hdr.level != 0);
  }
}
void btree::recover(PMEMobjpool *pool)
{
  pop = pool;
  version++;
  cout << "new version : " << version << endl;
}
