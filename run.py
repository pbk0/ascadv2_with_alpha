import pathlib
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras as ke

# Path for `Ascad_v2_dataset_extracted.h5` dataset file downloaded from
# https://zenodo.org/records/7885814/files/Ascad_v2_dataset_extracted.h5
_DATASET_FILE = "/mnt/d/dnd/Download/sca.ascad_v2_mo/Dn/file"

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size):
    x = inputs_core
    for block in range(convolution_blocks):
        x = ke.layers.Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same')(x)
        x = ke.layers.BatchNormalization()(x)
        x = ke.layers.AveragePooling1D(pool_size = pooling_size)(x)
    
    output_layer = ke.layers.Flatten()(x)

    return output_layer


def dense_core(inputs_core,dense_blocks,dense_units,activated = False):
    x = inputs_core
    
    for block in range(dense_blocks):
        x = ke.layers.Dense(dense_units, activation='selu')(x)
        x = ke.layers.BatchNormalization()(x)
        
    if activated:
        output_layer = ke.layers.Dense(256,activation ='softmax' )(x)
    else:
        output_layer = ke.layers.Dense(256)(x)
    return output_layer


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
    return tf.data.Dataset.from_tensor_slices((
        {
            "trace_rin": _data[0]['inputs_rin'][..., 0],
            "trace_intermediate": _data[0]['inputs_intermediate'][..., 0],
            "alpha": _data[0]['alpha'],
            # "alpha": _data[0]['alpha']+np.random.randint(0, 255, size=_data[0]['alpha'].shape),
        },
        _data[1]['output'],
    )).batch(_batch_size)


def build_model() -> ke.Model:
    
    # make inputs
    _input_trace_rin = ke.layers.Input(shape=(1605, 1), name='trace_rin')
    _input_trace_intermediate = ke.layers.Input(shape=(93, 1), name='trace_intermediate')
    _input_alpha = ke.layers.Input(shape=(1, ), name='alpha')
    
    # layers for _input_trace_rin
    _input_trace_rin_x = cnn_core(_input_trace_rin, convolution_blocks=1, kernel_size=[32],
                           filters=16, strides=10, pooling_size=2)
    _input_trace_rin_x = ke.layers.Dense(300, activation='selu')(_input_trace_rin_x)
    _input_trace_rin_x = ke.layers.BatchNormalization()(_input_trace_rin_x)
    _input_trace_rin_x = ke.layers.Dense(128, activation='selu')(_input_trace_rin_x)
    _input_trace_rin_x = ke.layers.BatchNormalization()(_input_trace_rin_x)
    
    # layers for _input_trace_intermediate
    _input_trace_intermediate_x = cnn_core(
        _input_trace_intermediate,
        convolution_blocks=2, kernel_size=[32, 8], filters=12, strides=1, pooling_size=2
    )
    _input_trace_intermediate_x = ke.layers.Dense(100, activation='selu')(_input_trace_intermediate_x)
    _input_trace_intermediate_x = ke.layers.BatchNormalization()(_input_trace_intermediate_x)
    _input_trace_intermediate_x = ke.layers.Dense(32, activation='selu')(_input_trace_intermediate_x)
    _input_trace_intermediate_x = ke.layers.BatchNormalization()(_input_trace_intermediate_x)
    
    # make alpha embedding
    _embedding_alpha = ke.layers.Embedding(input_dim=256, output_dim=16)(_input_alpha)
    _embedding_alpha = ke.layers.Flatten()(_embedding_alpha)
    
    # let's concatenate three components
    _x = ke.layers.concatenate([_input_trace_rin_x, _input_trace_intermediate_x, _embedding_alpha])
    
    # let's make dense layers
    _x = ke.layers.Dense(256, activation='selu')(_x)
    _x = ke.layers.BatchNormalization()(_x)
    _x = ke.layers.Dense(200, activation='selu')(_x)
    _x = ke.layers.BatchNormalization()(_x)
    _x = ke.layers.Dense(200, activation='selu')(_x)
    _x = ke.layers.BatchNormalization()(_x)
    _x = ke.layers.Dense(200, activation='selu')(_x)
    _x = ke.layers.BatchNormalization()(_x)
    
    # now let's add softmax layer
    _output = ke.layers.Dense(256, activation='softmax')(_x)
    
    # let's make model
    _model = ke.Model(
        inputs={
            'trace_rin': _input_trace_rin,
            'trace_intermediate': _input_trace_intermediate,
            'alpha': _input_alpha,
        },
        outputs=_output
    )
    
    # compile the model
    _model.compile(loss='sparse_categorical_crossentropy', optimizer=ke.optimizers.Adam(0.0001), metrics=['accuracy'])
    
    # print the model summary
    _model.summary()
    
    # return the model
    return _model
    

def main():

    # ------------------------------------------------------- 01
    # fit the model
    _ret1 = load_dataset(
        byte=2, n_traces=200000, n_traces_val=50000, dataset='training'
    )
    _train_ds = make_tf_dataset(_ret1[0], 128)
    _validate_ds = make_tf_dataset(_ret1[1], 128)
    
    _model = build_model()
    _model.fit(_train_ds, validation_data=_validate_ds, epochs=100)

    # ------------------------------------------------------- 02
    # attack
    _ret2 = load_dataset(
        byte=2, n_traces=5000, dataset='attack'
    )
    _attack_ds = make_tf_dataset(_ret2, 128)
    
    # print(_ret2)



if __name__ == '__main__':
    main()