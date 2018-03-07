usage:
- create environment and install dependencies

using conda
```
conda create -n pupil python=3.6
source activate pupil
conda install pip
pip install -r requirements.txt
```

- run *Voight_Kampff_test.py* 
```
python Voight_Kampff_test.py -h
usage: Voight_Kampff_test.py [-h] [-f filename] [-o output_name]

let the madness begin!

optional arguments:
  -h, --help      show this help message and exit
  -f filename     victim filename (example: toy_data/data.mj2)
  -o output_name  name of files to store outputs (example: `output` will
                  produce 2 files `output_area.npy` and `output_center.npy`)

```

- in the opened window select region of interest and press *space*
- in the next window adjust kernel and threshold values untill you achieve enlightenment, press *q* key when you enter nirvana
- it will create *output_name*_area.npy and *output_name*_center.npy with numpy arrays containing results


TODO:
- povtikat in optical flow and stabilize w.r.t. reference point
- implement confidence score p ~ diff(pupil_size)
- keep modularity in mind (e.g. make smt like pipeline in scikit-learn)
- more verbosity
- optimization (resize, threading, kimchi-picking)
- ...
- profit
