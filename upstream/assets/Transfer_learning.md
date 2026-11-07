# Fine-tuning

Here are the details of the fine-tuning script. To do the model fine-tuning, you need to follow the steps below:

1. Change the `args_path` parameter to the checkpoint path of your pretrained model, **and remember to add a `/` at the end** to tell the computer to search the file under this path. (For example, if you want to fine-tune the CPMLP pretrained in the Li dataset to the Zn dataset, you should put the path of the pretrained CPMLP model checkpoint as: /path/to/your/checkpoint/.)
2. Change the `finetune_dataset` parameter to your target dataset.
3. Ensure the `num_process` parameter equals the number of GPUs you want to use. (You can use the `CUDA_VISIBLE_DEVICES=0,1` to set the GPUs index of 0 and 1 to do the fine-tuning, so the `num_process=2` here.)
4. Run the command to do the fine-tuning.



# Domain adaptation

Here are the details of the domain adaptation script. To do the model domain adaptation, you need to follow the steps below:

1. Change the `dataset` parameter to the dataset you want to set as the source dataset. (For example, if you want to do the domain adaptation from the Li dataset to the Na dataset, you should input `dataset=MIX_large` here.)
2. Change the `target_dataset` parameter to your target dataset.
3. Ensure the `num_process` parameter equals the number of GPUs you want to use.
4. Run the command to do the domain adaptation.