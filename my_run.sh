PRE_RULE="champsim_print_test"
OUTPUT_NAME="training_ds_milc.txt"
WARM_UP=1
SIM_INS=200000000
TRACE_NAME="433.milc-127B.champsimtrace.xz"

touch ./results/$OUTPUT_NAME
#bin/$PRE_RULE --warmup_instructions $WARM_UP --simulation_instructions $SIM_INS ./traces/$TRACE_NAME > ./results/$OUTPUT_NAME | python ./prefetcher/py_co_test/py_co_test.py
bin/$PRE_RULE --warmup_instructions $WARM_UP --simulation_instructions $SIM_INS ./traces/$TRACE_NAME > ./results/$OUTPUT_NAME
cat ./results/$OUTPUT_NAME
echo "WRITE DONE\n\n"
