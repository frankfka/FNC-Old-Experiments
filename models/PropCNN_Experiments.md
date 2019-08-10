# Experiment Log (PropCNN)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Agree, Disagree, Discuss) Left-to-Right/Up-to-Down

### Log 1 - Aug 8
#### Parameters

**Data:** Unbalanced data (agree, disagree, discuss): (3678, 840, 8909) with sequence length 500

**Model:** 
* Similar to TwoToOneLSTM, except inputs are 1 CNN + Dropout + Maxpool
* Concat -> Flatten -> 2 Dense -> Output Dense

**Params:**
```python
# Model Params
NUM_CLASSES = 3
SEQ_LEN = 500
EMB_DIM = 300
INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
DROPOUT = 0.5
NUM_CONV_HIDDEN = 64
KERNEL_SIZE_CONV = 3
USE_MAXPOOL = True
POOL_SIZE = 2
NUM_DENSE_HIDDEN = 512
# Optimizer
ADAM_LR = 0.001
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPSILON = 1e-08

# Training Params
NUM_EPOCHS = 30
BATCH_SIZE = 64
TRAIN_VAL_SPLIT = 0.2
```

#### Results

* Max validation accuracy at last epoch at 0.9713
* QUICK LEARNING RATE: Flattened by epoch 20
* Validation loss seemed to bottom out at epoch 20
* Overfitting seemed to be apparent at epoch 5: val loss > train loss
* Tested on *entire* test set (1903, 697, 4464):
    * Accuracy: 0.6142412231030577
    * F1 Score (Macro): 0.4070038374018583
    * F1 Score (Micro): 0.6142412231030577
    * F1 Score (Weighted): 0.5938877357432255
    * Confusion matrix:
    ```
    [[0.57172885 0.00210194 0.42616921]
     [0.56384505 0.00573888 0.43041607]
     [0.27105735 0.0015681  0.72737455]]
    ```