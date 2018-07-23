# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program.

The main project deliberates are:

- `Image Classifier Project.ipynb` Jupyter Notebook
- `Image Classifier Project.html` HTML export of the Jupyter Notebook above.
- `train.py` to train a new network on a data set.
- `predict.py` to predict flower name from an image.


## Assets

Image categories are found in `cat_to_name.json` and flower images can be downloaded in the gziped tar file [flower_data.tar.gz] from Udacity.

Get flower flowers:
```bash
mkdir -p flowers
cd data
wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
tar xzvf ./flower_data.tar.gz
```

You should now have **test**, **train** and **valid** directories containing classification directories and flower images under the **flowers** directory.

## Examples

Train on **CPU** with default **vgg16**:
```bash
python ./train.py ./flowers/train/
```

Train on **GPU** with **densenet121** with one **500** node layer:
```bash
python ./train.py ./flowers/train --gpu --arch "densenet121" --hidden_units 500 --epochs5
```

Additional hidden layers with checkpoint saved to densenet201 directory.
```bash
python ./train.py ./flowers/train --gpu --arch=densenet201 --hidden_units 1280 640 --save_dir densenet201
```

## Part 1

### [Image Classifier Project.ipynb]

To review the  [Image Classifier Project.ipynb] notebook, launch **Jupyter Notebook** from the project root:

```bash
jupyter notebook
```

## Part 2

### [train.py]

**Options:**

- Set directory to save checkpoints
    - `python train.py data_dir --save_dir save_directory`
- Choose architecture
    - `python train.py data_dir --arch "vgg13"`
- Set hyperparameters
    - `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
- Use GPU for training
    - `python train.py data_dir --gpu`

**Help** - `python ./train.py -h`:
```plain
usage: python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 3136 --epochs 5

Train and save an image classification

positional arguments:
  data_directory

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory to save training checkpoint file (default:
                        .)
  --save_name SAVE_NAME
                        Checkpoint filename. (default: checkpoint)
  --categories_json CATEGORIES_JSON
                        Path to file containing the categories. (default:
                        cat_to_name.json)
  --arch ARCH           Supported architectures: vgg11, vgg13, vgg16, vgg19,
                        densenet121, densenet169, densenet161, densenet201
                        (default: vgg16)
  --gpu                 Use GPU (default: False)

hyperparameters:
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --hidden_units HIDDEN_UNITS [HIDDEN_UNITS ...], -hu HIDDEN_UNITS [HIDDEN_UNITS ...]
                        Hidden layer units (default: [3136, 784])
  --epochs EPOCHS       Epochs (default: 1)
```

### [predict.py]

- Basic usage
    - `python predict.py /path/to/image checkpoint`
- Options
    - Return top KK most likely classes
        - `python predict.py input checkpoint --top_k 3`
    - Use a mapping of categories to real name
        - `python predict.py input checkpoint --category_names cat_to_name.json`
    - Use GPU for inference
        - `python predict.py input checkpoint --gpu`

[flower_data.tar.gz]:https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
[Image Classifier Project.ipynb]:https://github.com/cjimti/personal-aipnd-project/blob/master/Image%20Classifier%20Project.ipynb
[train.py]:https://github.com/cjimti/personal-aipnd-project/blob/master/train.py
[predict.py]:https://github.com/cjimti/personal-aipnd-project/blob/master/predict.py
