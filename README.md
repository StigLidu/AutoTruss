# AutoTruss: Automatic Truss Design with Reinforcement Learning

The official repo of paper 'Automatic Truss Design with Reinforcement Learning'

## Generated Truss

The generated trusses are in folder 'assets', you can evaluate them by following command:

```
python apps/eval.py --config $TRUSS_CONFIG$ --draw-file $TRUSS_PATH$
```

For example:

```
python apps/eval.py --config 17_bar_case --draw-file assets/17_bar_1378.txt
```

This command will generate a truss image.

## Run Experiment:

```
./run.sh without_buckle_case1 test
```

The command will generate truss automatically and save the truss in 'PostResults/without_buckle_case1/test/'

Or use the following instructions sequentially:

### Stage 1

Generating truss layouts in stage 1.

```
python ./Stage1/continuous_uct_3d.py --config without_buckle_case1 --run-id test
```

Change output's format to input for stage 2.

```
python ./Stage1/noise_input_permutation_format_transfer.py --config without_buckle_case1 --run-id test
```

### Stage 2

Use RL to search for lighter truss layouts.

```
python ./Stage2/main_3d.py --config without_buckle_case1 --run-id test
```

## Add Costomized Config

To generate your own truss with different configs, you can add your own config file in 'configs' and register it in 'config.py'.
