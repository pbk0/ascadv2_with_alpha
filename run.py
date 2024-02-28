import pathlib
import numpy as np
import h5py
import tensorflow as tf

# Path for `Ascad_v2_dataset_extracted.h5` dataset file downloaded from
# https://zenodo.org/records/7885814/files/Ascad_v2_dataset_extracted.h5
_DATASET_FILE = "/mnt/d/dnd/Download/sca.ascad_v2_mo/Dn/file"


def read_from_h5_file(n_traces=1000, dataset='training', load_plaintexts=False):
    f = h5py.File(_DATASET_FILE, 'r')[dataset]
    labels_dict = {
        _k: _v[:n_traces] for _k, _v in f['labels'].items()
    }
    if load_plaintexts:
        data = {'keys': f['keys'][:n_traces], 'plaintexts': f['plaintexts'][:n_traces]}
        return f['traces'][:n_traces], labels_dict, data
    else:
        return f['traces'][:n_traces], labels_dict
    
def load_dataset(byte, flat=False, whole=False, n_traces=None, n_traces_val=10000,
                 dataset='training', print_logs=True):
    """
    Adapted from here
      https://github.com/sca-research/multi-vs-single-experiments/blob/f14da9cb1907dd5ed0d48ea81521585e1d07d056/ascadv2/utility.py#L261
    """
    
    target = 't1'
    training = dataset == 'training'
    if print_logs:
        str_targets = 'Loading samples and labels for {}'.format(target)
        print(str_targets)
    
    traces, labels_dict = read_from_h5_file(n_traces=n_traces, dataset=dataset)
    traces = np.expand_dims(traces, 2)
    X_profiling_dict = {}
    
    alpha = np.array(labels_dict['alpha'], dtype=np.uint8)[:n_traces]
    X_profiling_dict['alpha'] = alpha
    if whole:
        X_profiling_dict['traces'] = traces[:, 4088:4088 + 1605 + 93 * 16]
    
    elif flat:
        intermediate_points = traces[:, 4088 + 1605:4088 + 1605 + 93 * 16]
        X_profiling_dict['inputs_intermediate'] = intermediate_points
        X_profiling_dict['inputs_rin'] = traces[:, 4088:4088 + 1605]
    else:
        X_profiling_dict['inputs_intermediate'] = traces[:,
                                                  4088 + 1605 + 93 * byte:4088 + 1605 + 93 * (
                                                              byte + 1)]
        X_profiling_dict['inputs_rin'] = traces[:, 4088:4088 + 1605]
    
    if training:
        traces_val, labels_dict_val = read_from_h5_file(n_traces=n_traces_val,
                                                        dataset='validation')
        traces_val = np.expand_dims(traces_val, 2)
        X_validation_dict = {}
        
        alpha = np.array(labels_dict_val['alpha'], dtype=np.uint8)[:n_traces_val]
        X_validation_dict['alpha'] = alpha
        if whole:
            X_validation_dict['traces'] = traces_val[:,
                                          4088:4088 + 1605 + 93 * 16]
        
        elif flat:
            X_validation_dict['inputs_intermediate'] = traces_val[:,
                                                       4088 + 1605:4088 + 1605 + 93 * 16]
            X_validation_dict['inputs_rin'] = traces_val[:, 4088:4088 + 1605]
        else:
            X_validation_dict['inputs_intermediate'] = traces_val[:,
                                                       4088 + 1605 + 93 * byte:4088 + 1605 + 93 * (
                                                                   byte + 1)]
            X_validation_dict['inputs_rin'] = traces_val[:, 4088:4088 + 1605]
    
    Y_profiling_dict = {}
    
    permutations = np.array(labels_dict['p'], np.uint8)[:n_traces, byte]
    real_values_t1_temp = np.array(labels_dict[target], dtype=np.uint8)[
                          :n_traces]
    real_values_t1 = np.array([real_values_t1_temp[i, permutations[i]] for i in
                               range(len(real_values_t1_temp))])
    Y_profiling_dict['output'] = real_values_t1
    
    if training:
        Y_validation_dict = {}
        permutations_val = np.array(labels_dict_val['p'], np.uint8)[:n_traces_val,
                           byte]
        real_values_t1_temp_val = np.array(labels_dict_val[target],
                                           dtype=np.uint8)[:n_traces_val]
        real_values_t1_val = np.array(
            [real_values_t1_temp_val[i, permutations_val[i]] for i in
             range(len(real_values_t1_temp_val))])
        Y_validation_dict['output'] = real_values_t1_val
        return (X_profiling_dict, Y_profiling_dict), (X_validation_dict, Y_validation_dict)
    
    else:
        return (X_profiling_dict, Y_profiling_dict)


def make_tf_dataset(_data, _batch_size) -> tf.data.Dataset:
    _trace = np.concatenate(
        [
            _data[0]['inputs_rin'][..., 0],
            _data[0]['inputs_intermediate'][..., 0],
        ],
        axis=1
    )
    return tf.data.Dataset.from_tensor_slices((
        {
            "trace": _trace,
            "alpha": _data[0]['alpha'],
        },
        _data[1]['output'],
    )).batch(_batch_size)

def main():

    # ------------------------------------------------------- 01
    # fit the model
    _ret1 = load_dataset(
        byte=2, n_traces=20000, n_traces_val=1000, dataset='training'
    )
    _train_ds = make_tf_dataset(_ret1[0], 128)
    _validate_ds = make_tf_dataset(_ret1[1], 128)

    # ------------------------------------------------------- 02
    # attack
    _ret2 = load_dataset(
        byte=2, n_traces=5000, dataset='attack'
    )
    _attack_ds = make_tf_dataset(_ret2, 128)
    
    print(_ret2)



if __name__ == '__main__':
    main()