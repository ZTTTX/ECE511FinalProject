
To make other prefetch excutables:
1. change output file name in ./champsim_config.json, line 2
2. change config in ./champsim_config.json

3. in shell, run: ./config.sh champsim_config.json
4. in shell, run: make
Output executable binary will be at ./bin

Trace will be place in ./traces 

To run simulation:
1. open ./my_run.sh
2. variables:
	PRE_RULE: the binary executable file used in the simulation
	OUTPUT_NAME: the result will be stored at ./resutls/OUTPUT_NAME
	WARM_UP: execlude the first ? cycles
	SIM_INS: simulation will run over ? cycles, after execluding the warmups
	TRACE_NAME: which trace to use
3.in shell, run:$ sh my_run.sh
To generate training data, run train_conv.py
 Results will be stored in train.o
 