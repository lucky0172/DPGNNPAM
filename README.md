# DPGNNPAM
A deep learning model that integrates path aggregation strategies and prototype-driven mechanisms for effective node classification on biological networks. DPGNNPAM captures both topological and attribute information, improving the prediction of drought-resistant genes in rice.

Below is the algorithm flowchart:

![Algorithm-Flowchart](https://github.com/lucky0172/DPGNNPAM/blob/91751cddbee278d013b07a8ef5d63052362e75b5/result/Algorithm.png)



## 💻 Installation

 To run DPGNNPAM, you need to set up a Python environment with the following core dependencies:

- Python ≥ 3.8 
- PyTorch (with optional CUDA support)
- PyTorch Geometric (PyG) for graph neural networks
- NumPy, Pandas, Scikit-learn for data processing

 We recommend using `conda` and `pip` to manage the environment.

### Step 1: Install PyTorch

 First, install **PyTorch** based on your system and hardware. Visit [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/) for the correct command. 

For example, for **CUDA 11.8**: 

#### Using pip

```bash 
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

#### Or using conda

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 2: Install PyTorch Geometric and Related Packages

```
# Replace {TORCH_VERSION} with your PyTorch version, e.g., torch-2.0.0
# Replace +cu118 with +cpu if using CPU-only
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Step 3: Install Other Dependencies

```
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

## 🚀 Training Process

Use the `main.py` script to train DPGNNPAM with customizable hyperparameters:

```
python main.py \ 
	--node_file_path node_file \
	--edge_file_path edge_file \
	--encoder DPGNNPAM \
	--runs 10 \
    --dropout 0.5 \
    --lr 0.01 \ 
    --weight_decay 5e-4 \ 
    --epochs 200
```

Parameter Description

- `--node_file_path`: Path to node feature file (replace with your node data file)
- `--edge_file_path`: Path to edge relation file (replace with your edge data file)
- `--encoder`: Encoder type, choose from: DPGNNPAM, GCN, GAT, GAE
- `--runs`: Number of experimental repetitions, 10 runs for averaged results
- `--dropout`: Dropout rate, set to 0.5 to prevent overfitting
- `--lr`: Learning rate, set to 0.01
- `--weight_decay`: Weight decay, L2 regularization coefficient 5e-4
- `--epochs`: Training epochs, 200 epochs

## 💾  Pretrained Weights

To facilitate quick testing or reproduction of results, we also provide **pretrained model weights**.
 These files are located in the `weights/` directory and can be directly loaded before running prediction.

When you run the prediction script (e.g. `python predict.py --encoder DPGNNPAM`),
 the program automatically loads the corresponding pretrained weights according to the selected encoder type.
 If you have saved the pretrained weights in a different directory, please modify the file paths in the script accordingly (for example, update the lines loading `.pkl` files to match your own path).

> 💡 **Note:** Make sure the weight filenames and encoder names are consistent, otherwise the model will fail to load correctly.

