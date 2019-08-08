# Experiment Log (TwoToOneLSTM)
Experiments are in reverse chronological order (most recent first)

*Notes:*
* Confusion matrix: Y-Axis is true, X-Axis is predicted. Label Order: (Agree, Disagree, Discuss) Left-to-Right/Up-to-Down

### Log 3 - Aug 7
#### Parameters

**Data:** Unbalanced data (agree, disagree, discuss): (3678, 840, 8909) with sequence length 500

**Model:** Same as previous logs, but using Bidirectional LSTM

**Params:** 
```python
# Model Params
SEQ_LEN = 500
EMB_DIM = 300
INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
DROPOUT = 0.5
NUM_LSTM_UNITS = 64
LSTM_BIDIRECTIONAL = True
NUM_DENSE_HIDDEN = 512
NUM_CLASSES = 3
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

* Max validation accuracy at last epoch at 0.9542, did not increase much from epoch 25
* Validation loss bottomed out by epoch 25
* Tested on *entire* test set (1903, 697, 4464):
    * Accuracy: 0.6162231030577576
    * F1 Score (Macro): 0.4547077214904622
    * F1 Score (Micro): 0.6162231030577576
    * F1 Score (Weighted): 0.613119716896711
    * Confusion matrix:
    ```
    [[0.6001051  0.05149764 0.34839727]
     [0.55954089 0.09325681 0.3472023 ]
     [0.26456093 0.03068996 0.7047491 ]]
    ```

### Log 2 - Aug 6
#### Parameters

**Data:** Unbalanced data (agree, disagree, discuss): (3678, 840, 8909) with sequence length 500

**Model:** Same as Log 1

**Params:** Same as Log 1

#### Results

* Max validation accuracy at last epoch at 0.74
* Validation loss began increasing at epoch 15
* Tested on *entire* test set (1903, 697, 4464):
    * Accuracy: 0.5501132502831257
    * F1 Score: 0.5501132502831257
    * Confusion matrix:
    ```
    [[0.6037835  0.0903836  0.3058329]
     [0.56527977 0.15494978 0.27977044]
     [0.33691756 0.07414875 0.58893369]]
    ```

### Log 1 - Aug 6
#### Parameters

**Data:** Balanced data (agree, disagree, discuss): (840, 840, 840) with sequence length 500

**Model:** 
Two separate input LSTM -> Concat -> Batch Norm -> Dense -> Dense -> Dense Output

**Params:**
```python
# Model Params
SEQ_LEN = 500
EMB_DIM = 300
INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
DROPOUT = 0.5
NUM_LSTM_UNITS = 64
NUM_DENSE_HIDDEN = 512
NUM_CLASSES = 3

# Training Params
NUM_EPOCHS = 30
BATCH_SIZE = 64
TRAIN_VAL_SPLIT = 0.2
```

#### Results

* Max validation accuracy at last epoch at 0.88
* Validation loss and accuracy continued to improve, can potentially train longer
* Tested on *entire* test set (1903, 697, 4464):
    * Accuracy: 0.5501132502831257
    * F1 Score: 0.5501132502831257
    * Confusion matrix:
    ```
    [[0.63636364 0.0189175  0.34471886]
     [0.60832138 0.02439024 0.36728838]
     [0.29278674 0.01836918 0.68884409]]
    ```