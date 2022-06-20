# CS420-Final-Project

Transform data to png formd data:

```shell
python main.py -e exp_specs/seq2png.yaml
python main.py -e exp_specs/png2npz.yaml
cd tools
python quickdraw_visual_to_hdf5.py
```

Run experiment example

```shell
python main.py -e exp_specs/swin_r2cnn.yaml -g 0 # g is the gpu number
```

