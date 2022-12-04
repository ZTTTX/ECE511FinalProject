PRE_RULE="champsim_print_test"
OUTPUT_NAME="output_astar_ipstride.txt"
WARM_UP=10000
SIM_INS=2000000
TRACE_NAME="473.astar-153B.champsimtrace.xz"

touch ./results/$OUTPUT_NAME
#bin/$PRE_RULE --warmup_instructions $WARM_UP --simulation_instructions $SIM_INS ./traces/$TRACE_NAME > ./results/$OUTPUT_NAME | python ./prefetcher/py_co_test/py_co_test.py
bin/$PRE_RULE --warmup_instructions $WARM_UP --simulation_instructions $SIM_INS ./traces/$TRACE_NAME > ./results/$OUTPUT_NAME
cat ./results/$OUTPUT_NAME
echo "WRITE DONE\n\n"
