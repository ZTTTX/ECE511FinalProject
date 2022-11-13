# -*- coding: utf-8 -*-
import socket
import os
import sys
import time
from transformer_model import init_model, translate

def main():

    model = init_model()

    print(" ")
    print("============================")
    print("Server Started, Waiting...")
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM,0)
    ip_port = ('127.0.0.1',9998)
    sock.bind(ip_port)
    sock.listen(5)
    conn,address = sock.accept()
    num = 0
    print('starting socket', conn, ' ', address)

    while True:
        recv_data = conn.recv(1024)
        recv_data = recv_data.decode("ascii").rstrip('\x00')
        # if recv_data != '':
        #     print('python get:',recv_data)
        # time.sleep(0.01)
        # addr_out = 1684654016 
        recv_data = recv_data.split(",")
        try:
            recv_data = [int(i) for i in recv_data]
        except:
            print(f"Number of inferrences: {num}")
            num = 0
            conn.close()
            sock.listen(5)
            conn,address = sock.accept()
            continue
        # print("Length of recieved data: ", len(recv_data))

        # next-line prefetcher
        # addr = int(recv_data[-1])
        # addr_out = addr + (1 << 6)

        # transformer prefetcher

        addr_out = translate(model, recv_data)[3]
        num += 1
        print(f'current address: {recv_data[-1]}, transformer predicted: {addr_out}')

        send_data = str(addr_out) #.ljust(1024,'\n')
        conn.send(bytes(send_data, encoding="ascii"))

    conn.close()

if __name__ == "__main__":

    main()
