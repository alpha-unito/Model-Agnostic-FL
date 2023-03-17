# Distributed Boosting (and Bagging) examples

To run a tutorial, just open a terminal window in the `director` folder of the desired example and run
```
$ fx director start --disable-tls -nn False -c director.yaml
```
to set up the director.

Then, open a terminal window for each `envoy` folder provided by the tutorial (usually 10, in a few cases 2) and for each of them run
```
$ fx envoy start -n $ENVOY --disable-tls --envoy-config-path envoy_config.yaml -dh localhost -dp 50052
```
to set up the envoy. The `$ENVOY` variable contains the name of the envoy, and its choice is up to the user.

Finally, open a last terminal window in the `workspace` folder and simply run the experiment execution file; for example, in case of the AdaBoostF_Adult tutorial, the command is the following:
```
$ python AdaboostF_adult.py
```
The experiment execution file can be modified according to the user's needs; it is also possible to specify some parameter from command line, just run the execution file with the -h flag to check them.


DISCLAIMER: if you want to log the results into wandb the name assigned to each envoy should be something like `envoy_n`, where `n` is the envoy rank, and then in the openfl-extended/openfl/component/collaborator/Collaborator.py file and in the experiment execution file you should change the LOG_WANDB variable value into True.