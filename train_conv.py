
filename_i = 'results/training_ds_milc.txt'
filename_o = 'train_o/training_ds_milc_433.txt'

fi = open(filename_i,'r')
fo = open(filename_o,'w')

for line in fi.readlines():
    line = line.strip()
    if line !='':
        if line[0]=='!':
            print('running')
            fo.write(line[2:]+'\n')
            
print('run_done')      
fi.close()
fo.close()

