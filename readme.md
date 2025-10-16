
# A General Tabuler Diffusion Model
🧭 Guide
## Step 1
Make sure your dir tree looks like this:
```
project - diff_scripts
        - inference
        - model
        - priors 
        ...
```
Enter this dir and activate the conda env:
```python
$ cd project
$ conda activate PFN
```
Now you can prepare data, which will be used for training diffusion model in step 2. 
```python
$ python diff_scripts/preprocess.py --config /mnt/public/hxy/project/diff_scripts/config_quantile_uniform/preprocess_default.yaml
```

## Step 2

```python
cd diff_scripts
```

If this is the first time you train this repo, run 
```python
$ accelerate config
```
Now you can train the diffusion model
```
$ accelerate launch train_script.py experiment_name=exp_01 base_dir=/mnt/public/hxy/diff_data/quantile_normal
```
Notice that the `padding_to` in this config file should be the same as `padding_to` used in step 1. 


## Step 3
You have finished training your magic tabular diffusion model. 

It is time to test your model's perfomance. Let's use it to generate more samples for a dataset. 

```python
$ cd diff_script
```
```python
$ python sample_script.py --preprocess_config /mnt/public/hxy/project/diff_scripts/config_quantile_normal/preprocess_default.yaml --train_config /mnt/public/hxy/project/diff_scripts/config/default.yaml --benchmark_dir /mnt/public/cls/classifier_benchmarks/benchmark_504 --dataset_name water-quality --num_samples 20 --ckpt_path /mnt/public/hxy/project/diff_scripts/results/quantile_normal_steps_500000_random_batch/model-500000.pt --sample_save_dir /mnt/public/hxy/project/diff_scripts/results 

$ python sample_script.py --preprocess_config /mnt/public/hxy/project/diff_scripts/config_quantile_normal/preprocess_single_dataset.yaml --train_config /mnt/public/hxy/project/diff_scripts/config/default.yaml --benchmark_dir /mnt/public/cls/classifier_benchmarks/benchmark_504 --dataset_name water-quality --num_samples 10 --ckpt_path /mnt/public/hxy/project/diff_scripts/results/single_water_quality/model-1000.pt --sample_save_dir /mnt/public/hxy/project/diff_scripts/results

```