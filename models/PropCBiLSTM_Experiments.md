# Experiment Log (PropCBiLSTM)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Agree, Disagree, Discuss) Left-to-Right/Up-to-Down

### Log 2 - Aug 9
#### Parameters

**Data:** Unbalanced data (agree, disagree, discuss): (3678, 840, 8909) with sequence length 500

**Model:** 
* Similar to TwoToOneLSTM, except inputs are (1 CNN + Dropout + Maxpool) -> LSTM
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
NUM_LSTM_UNITS = 64
LSTM_BIDIRECTIONAL = True
LSTM_BEFORE_CNN = False
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

* Max validation accuracy at last epoch at 0.9769
* Quite quick to train, similar to Bi-LSTM model
* Validation loss continued to decrease through last epoch, trend increased above training loss around epoch 20, but not
extremely apparent (i.e. did not overtrain)
* Tested on *entire* test set (1903, 697, 4464):
    * Accuracy: Accuracy: 0.5981030577576444
    * F1 Score (Macro): 0.41660573986898486
    * F1 Score (Micro): 0.5981030577576444
    * F1 Score (Weighted): 0.5879574479296883
    * Confusion matrix:
    ```
    [[0.57330531 0.03152916 0.39516553]
     [0.56527977 0.03730273 0.3974175 ]
     [0.29413082 0.00963262 0.69623656]]
    ```

### Log 1 - Aug 8
#### Parameters

**Data:** Unbalanced data (agree, disagree, discuss): (3678, 840, 8909) with sequence length 500

**Model:** 
* Similar to TwoToOneLSTM, except inputs are LSTM + (1 CNN + Dropout + Maxpool)
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
NUM_LSTM_UNITS = 64
LSTM_BIDIRECTIONAL = True
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

* Max validation accuracy at last epoch at 0.9594
* SLOW learning rate. Slow training rate because of high number of parameters
* Validation loss seemed to bottom out at epoch 25
* Tested on *entire* test set (1903, 697, 4464):
    * Accuracy: 0.6201868629671574
    * F1 Score (Macro): 0.44207831641138573
    * F1 Score (Micro): 0.6201868629671574
    * F1 Score (Weighted): 0.6115465431908782
    * Confusion matrix:
    ```
    [[0.65370468 0.01366264 0.33263269]
     [0.68579627 0.04734577 0.26685796]
     [0.29480287 0.00985663 0.6953405]]
    ```