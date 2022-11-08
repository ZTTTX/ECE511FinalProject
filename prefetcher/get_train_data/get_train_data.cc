#include "cache.h"
#include "champsim.h"
#include <cstdio>

#define EXTRACT_PAGE_ID(addr)   ((addr) >> LOG2_PAGE_SIZE)              /* Extract the page ID */
#define EXTRACT_BLOCK_ID(addr)  (((addr) >> LOG2_BLOCK_SIZE) & 0x3f)    /* Extract the block ID within the page */

void CACHE::prefetcher_initialize() {}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in)
{ 
  uint64_t page_id = EXTRACT_PAGE_ID(addr);        /* Extract out the page ID of the current load/store */
  uint32_t block_id = EXTRACT_BLOCK_ID(addr);      /* Extract out the block ID of the current load/store */
  int32_t prefetch_block_id = (int32_t) block_id;  /* Temporarily store the block ID that we'll increment/decrement later on */
  uint64_t prefetch_addr;  
  printf("!,addr,%llu,page_id,%llu,block_id,%d,cache_hit,%hhu\n", addr, page_id, block_id, cache_hit);
  return metadata_in;
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}
