# membership_inference_attack
## Implementation of the paper : "Membership Inference Attacks Against Machine Learning Models", Shokri et al.

I implement the most basic attack which assumes the adversary has the data which comes from the same distribution as the target model’s the training dataset. I choose to evaluate on MNIST,
CIFAR10 and CIFAR100. I used the framework pytorch for the target and the shadow models and ligth gradient boosting for the attack model.

Tested on python 3.5 and torch 1.0

## Congiguration

in the file `config/config.yaml`, you will find the different settings you can change.

## Running experiements

By running main.py, you start the statistic proposed in `statistics.type` in the `config.yaml`. 
`training_size` will test all the values in `training_size_value`
`overfitting` will test all the values in `epoch_value`
`number_shadow` will test all the values in `number_shadow_value`
