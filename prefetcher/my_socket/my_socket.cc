#include "cache.h"
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <memory.h>
#include <signal.h>
#include <time.h>
#include<stdlib.h>
#include <arpa/inet.h>
#include <cstdio>
#include <iostream>
#include <cctype>
#include <algorithm>
#include<cstdlib>
#include <errno.h>
#include <queue>


int sockfd;
std::string addr_queue;
int init_count;
  // queue<uint64_t> addr_q;

void CACHE::prefetcher_initialize()
{
  init_count = 0;
  addr_queue = "0";

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
      perror("socket error");
      exit(1);
  }
  struct sockaddr_in serveraddr;
  memset(&serveraddr, 0, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  serveraddr.sin_port = htons(atoi("9998"));
  inet_pton(AF_INET, "127.0.0.1", &serveraddr.sin_addr.s_addr);
  if (connect(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0) {
      perror("connect error");    
      exit(1);
  }
}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in) 
{ 
  printf("[DEBUG] Addr: %llu\n", addr);
  if (init_count < 16){
    addr_queue = addr_queue +  "," + std::to_string(addr);
    init_count = init_count + 1;
  }
  else {
    int pos1 = addr_queue.find(",", 0, 1);
    addr_queue = addr_queue.substr(pos1 + 1, addr_queue.length());
    addr_queue = addr_queue + "," + std::to_string(addr);
  }
  if (init_count == 16){
    //SEND
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));
    strcpy(buffer, addr_queue.c_str());
    // printf("buffer is %s\n", buffer);
    // printf("input addresses: %s \n", addr_queue.c_str());
    if (write(sockfd, buffer, sizeof(buffer)) < 0) {
        perror("write error");  
    }
  
    //READ
    char str_o[1024];
    int len;
    if((len = recv(sockfd, str_o, 1024*sizeof(char), 0))<0)
      perror("recv");

    
    *std::remove(str_o, str_o+strlen(str_o), '\n') = '\0';
    uint64_t addr_out = strtoull(str_o, NULL, 0);
    // printf("[INFO] Current addr: %lu, Transformer predicted: %lu\n", addr, addr_out);
    prefetch_line(ip, addr, addr_out, true, 0);
  }
  return metadata_in; 
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}
