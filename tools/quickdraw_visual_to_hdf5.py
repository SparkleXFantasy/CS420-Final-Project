import h5py
import numpy as np
import os.path
from pathlib import Path
import pickle

import sys
import warnings
warnings.filterwarnings("ignore")

_project_folder_ = os.path.realpath(os.path.abspath('..'))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

# Arguments
quickdraw_root = '../../dataset/quickdraw_visual'
output_root = quickdraw_root

if not os.path.exists(output_root):
    os.makedirs(output_root)


def get_categories():
    res = [npz_file.stem[:-4] for npz_file in list(Path(quickdraw_root).glob('*.npz'))]
    return sorted(res, key=lambda s: s.lower())


def load_npz(file_path):
    npz = np.load(file_path, encoding='latin1', allow_pickle=True)
    return npz['train'], npz['valid'], npz['test']


def cvrt_category_to_points3(points3_arrays, hdf5_group=None):
    res = []
    for pts3_arr in points3_arrays:
        if hdf5_group is not None:
            hdf5_group.create_dataset(str(len(res)), data=pts3_arr)
        res.append(pts3_arr)
    print(len(res))
    return res


category_names = get_categories()
print('[*] Number of categories = {}'.format(len(category_names)))
print('[*] ------')
print(category_names)
print('[*] ------')

hdf5_names = ['train', 'valid', 'test']
mode_indices = [list() for hn in hdf5_names]
hdf5_files = [h5py.File(os.path.join(output_root, 'quickdraw_visual_{}.hdf5'.format(hn)), 'w', libver='latest') for hn in
              hdf5_names]
hdf5_groups = [h5.create_group('/sketch') for h5 in hdf5_files]

for cid, category_name in enumerate(category_names):
    print('[*] Processing {}th category: {}'.format(cid + 1, category_name))

    train_valid_test = load_npz(os.path.join(quickdraw_root, category_name + "_png" + '.npz'))

    for mid, mode in enumerate(hdf5_names):
        hdf5_category_group = hdf5_groups[mid].create_group(str(cid))
        pts3_arrays = cvrt_category_to_points3(train_valid_test[mid], hdf5_category_group)
        nsketches = len(pts3_arrays)

        hdf5_category_group.attrs['num_sketches'] = nsketches
        mode_indices[mid].extend(list(zip([cid] * nsketches, range(nsketches))))

for gid, gp in enumerate(hdf5_groups):
    gp.attrs['num_categories'] = len(category_names)

for hf in hdf5_files:
    hf.flush()
    hf.close()

pkl_save = {'categories': category_names, 'indices': mode_indices}
with open(os.path.join(output_root, 'categories.pkl'), 'wb') as fh:
    pickle.dump(pkl_save, fh, pickle.HIGHEST_PROTOCOL)

print('All done.')