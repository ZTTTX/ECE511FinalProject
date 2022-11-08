#include "cache.h"
#include "champsim.h"
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cctype>
#include <algorithm>
#include<cstdlib>
#include <errno.h>

void CACHE::prefetcher_initialize() {}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in) 
{ 
  //BODY OF PYTHON READ
  FILE *fp;
  char str_o[100]; 
  char cmd[100];
  uint64_t num;
  sprintf(cmd, "python -c 'import py_co_test; py_co_test.test(%llu)'", addr);
  // strcat(cmd, std::to_string(addr));
  // strcat(cmd, ")");
  fp = popen(cmd, "r");
  while (fgets(str_o, sizeof(str_o)-1, fp)!=NULL){
    //printf("==%s==",str_o);
    //printf("IN!\n");
  }
  *std::remove(str_o, str_o+strlen(str_o), '\n') = '\0';
  // printf("++%s++",str_o);
  uint64_t addr_out = strtoull(str_o, NULL, 0);
  // printf("--%llu--",addr_out);
  pclose(fp);
  //END OF PYTHON READ, RETURN addr_out IS THE PREFETCH DESICION
  uint64_t pf_addr = addr_out;
  prefetch_line(ip, addr, pf_addr, true, 0);
  return metadata_in; 
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}
