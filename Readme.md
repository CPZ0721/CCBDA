## CCBDA HW1 - Vedio action classification

## Dataset
- Unzip data.zip to `./`
    ```sh
    unzip data.zip -d ./
    ```
- Folder structure
    ```
    .
    ├── train/
    ├── test/
    ├── test.py
    ├── Readme.md
    ├── requirements.txt
    ├── train.py
    ├── data_preprocess.py
    ├── lr_scheduler.py
    └── model.py
    ```

## Environment

- Python 3.8.10
- CUDA 11.3
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python3 train.py
```

## Make Prediction
```sh
python3 test.py
```
The prediction file is `test_result.csv`.