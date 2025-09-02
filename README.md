# DICP: Deep In-Context Prompt for Event Causality Identification

Source code for EMNLP 2025 paper "**DICP: Deep In-Context Prompt for Event Causality Identification**"

## Requirements

```shell
conda create -n DICP python=3.8.12
conda activate DICP
```

```shell
pip install -r requirements.txt
```
## Datasets

Download and unzip datasets, put them in `/data/raw_data`.

- EventStoryLine
  https://github.com/tommasoc80/EventStoryLine
- Causal-TimeBank
  https://github.com/paramitamirza/Causal-TimeBank

## Data Preprocess

### Step 1: Read documents
```
python read_document.py
python generate_sample.py
```

### Step 2: AMR parsing

Use the pre-trained AMR parser [parse_xfm_bart_large v0.1.0](https://github.com/bjascob/amrlib) parsing the data.

```
python get_amr_data.py
```

### Step 3: Prepare data

Prepare the data that the model used.

```
/preprocess/prepare_data.sh
```

## Running the model

Run the model.

```
/src/run.sh
```

When the program is finished, look at the log to get the final results.

#### arguments description:

in file /src/arguments.py:

    plm_path：pretrained language model path

    save_model_name：model save name

    device_num：device number

    fold：cross-validate data number
