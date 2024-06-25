# YOLOv8 Upgrade and AimBot

The detailed report in `Report.pdf` in the root dir

Model config file and codes in **[ultralytics/cfg/models](ultralytics/cfg/models)** and **[ultralytics/nn](ultralytics/nn)**, Shuffle Attention implemented in **[ultralytics/nn/modules/shuffleattention.py](ultralytics/nn/modules/shuffleattention.py)**

**`parse_model` function in `tasks.py` is changed to parse the new component**

All the scale used is `n` which will make `channels` to `0.25 * channels` 

You can just modify the `n` in`yolov8n-**` in train files to use more powerful models which requires hardware supports, especially a strong enough GPU.

--- 
Requirements:
- opencv-python~=4.10.0.82
- torch~=2.3.0+cu121
- numpy~=1.26.4
- pandas~=2.1.4
- pywin32~=305
- einops~=0.8.0
- pathlib~=1.0.1
- pillow~=10.2.0
- requests~=2.31.0
- psutil~=5.9.0
- streamlit~=1.30.0
- matplotlib~=3.8.0
- tqdm~=4.65.0
- yaml~=0.2.5
- pyyaml~=6.0.1
- scipy~=1.11.4
- future~=0.18.3
- pygetwindow~=0.0.9
- bettercam~=1.0.0
---

> Better to install pytorch through its website [Pytorch](https://pytorch.org/get-started/locally/) to install the version correspond to your hardware

### Use the commandline to install the requirements:

```shell
pip install -r requirements.txt
```

> There are example models that were trained in [models/*](models), you can use them directly

## Original YOLOv8 Network

Train YOLOv8:

```shell
python3 train_yolov8.py
```

## YOLOv8 with Shuffle-Attention (YOLOv8-SA)

### YOLOv8-SA1:

Train YOLOv8-SA1:

Config the `DATASET`, `epochs`, `batch` in the corresponding train python file to custom the training.

```shell
python3 train_yolov8sa1.py
```


### YOLOv8-SA3:

Train YOLOv8-SA3:

Config the `DATASET`, `epochs`, `batch` in the corresponding train python file to custom the training.

```shell
python3 train_yolov8sa3.py
```

## YOLOv8 with 4 Detect Head (YOLOv8-4Detect)

Train YOLOv8-4Detect:

Config the `DATASET`, `epochs`, `batch` in the corresponding train python file to custom the training.

```shell
python3 train_yolov8_4detect.py
```

## Evaluation

Config the `eval.py` first to specify the model and dataset(yaml path and the part like `val` or `test`)

Run the command:

```shell
python3 eval.py
```

to get the evaluation of the model.

## AimBot (Applying the Model)

To train the model, use the dataset in `dataset/data.yaml`

- The data from [Roboflow/Counter Strike 3](https://universe.roboflow.com/my-projects-qc3c9/counter-strike-3)

To apply the model, an AimBot for Counter Strike 2 was implemented

To use the AimBot, you firstly need to config the `config.py` file

The most important config is the `model`, which is the `pt` file path of the model you want to use

It is not recommend to change the `SCAN_REGION_WIDTH` and `SCAN_REGION_HEIGHT` which decide the region your model can see, unless you have powerful GPU that can support the calculation of larger size image.

`CONFIDENCE_THRESHOLD` is the least confidence of the result that are believed to be a target.

`QUIT_KEY` is the button to end the procedure, the default config is `Q` which means when `Q` is pressed during the procedure, it will exit immediately.

`MODE` is used to select the target you want to aim to.

`MOUSE_MOVE_RATE` is used to adjust your mouse rate, the lower means your mouse will move more slow but smoothly.

`HEAD_SHOT_MODE` is to set whether to aim the head of target

`VISUAL` is to set whether to show the view of model

After config complete, open your Counter Strike 2 and then start the procedure with:

```shell
python3 main.py
```

Wait for the procedure to start and use `Caps Lock` to control the aim

When `Caps Lock` is on, the procedure will aim to the target as config

When `Caps Lock` is off, the procedure will just view the region but will not move your mouse

Use `QUIT_KEY` (original set to `Q`) at any time to exit the procedure.
