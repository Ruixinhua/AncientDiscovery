# Ancient Discovery
This project includes experiment in paper: _A Novel Perspective to Decipher Oracle Bone Inscriptions: Artificial Intelligence_
The aim of this project is to present a method to decipher unknown Oracle bone inscriptions by using deep learning model

### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- CUDA enabled computing device(for fast calculating)

### Installation
```
$ git clone https://github.com/Ruixinhua/AncientDiscovery.git
$ cd AncientDiscovery
$ pip install -r requirements.txt
```

### Datasets
Download dataset in [Mendeley](https://data.mendeley.com/datasets/ksk47h2hsh/2).

### Usage
```
$ cd AncientDiscovery/experiment
$ python run.py -c ../configs/<config-file-name.yaml>
```

For 10-fold cross validation:
```
$ python cross_validation.py -r ../datasets/
```

It may takes a very long time to finalise all cross validation experiments, 
but it is possible to make predictions on the existing experiment model.
By executing the code:
```
$ python cross_prediction.py -r ../datasets/ -s 600
```

