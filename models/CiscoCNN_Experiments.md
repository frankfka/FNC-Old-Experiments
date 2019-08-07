# Experiment Log (CiscoCNN)
Experiments are in reverse chronological order (most recent first)

### Log 1 - Aug 6
#### Parameters

**Data:** Balanced data (agree, disagree, discuss): (840, 840, 840) with sequence length 1000

**Model:** Standard model highlighted in the Cisco repo

**Params:**
```python
# Model Params
NUM_CLASSES = 3
SEQ_LEN = 1000
EMB_DIM = 300
INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
DROPOUT = 0.5
NUM_CONV_HIDDEN = 256
KERNEL_SIZE_CONV = 3
USE_MAXPOOL = True
POOL_SIZE = 2
NUM_DENSE_HIDDEN = 1024
# Optimizer - default Adam

# Training Params
NUM_EPOCHS = 30
BATCH_SIZE = 64
TRAIN_VAL_SPLIT = 0.2
```

#### Results

* Max validation accuracy at last epoch at 0.58
* Validation loss began increasing at epoch 20
* Tested on *entire* test set (1903, 697, 4464):
    * Accuracy: 0.4421007927519819
    * F1 Score: 0.44210079275198194
    * Confusion matrix:
    ```
    [[0.57593274 0.29269574 0.13137152]
    [0.53945481 0.32568149 0.1348637 ]
    [0.35170251 0.24507168 0.40322581]]
    ```