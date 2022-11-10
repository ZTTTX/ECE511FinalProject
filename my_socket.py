# -*- coding: utf-8 -*-
import socket
import os
import sys
import time
#创建一个socket对象
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
    if recv_data != '':
        print('python get:',recv_data)
    time.sleep(0.01)   
    
    addr_out = 1684654016
    send_data = str(addr_out)
    conn.send(bytes(send_data, encoding="ascii"))

conn.close()