<p align="center">
  <h1 align="center"> ChampSim For ECE511 Final Project </h1>
  <p> This is a modified version of Champsim, aimed to work with python prefetchers <p>
</p>

# Origin
```
Origin: https://github.com/ChampSim/ChampSim.git
```

# Run

To modify hardware settings, change ./champsim_config.json
To make, run $ ./config.sh champsim_config.json
         and $ make

To modify simulation settings, change ./my_run.sh

To run simulation, run $ sh my_run.sh

Results are collected in ./results


# Python Prefetcher

It used a c++ file in ./prefetcher/py_co_test.cc to evoke the python prefetcher.  
To use a python prefetcher, place your prefetcher.py in ./ 
Change the popen() parameters to link your python prefetcher with the .cc file. 


