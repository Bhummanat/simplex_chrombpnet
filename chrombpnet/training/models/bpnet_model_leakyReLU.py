import numpy as np
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import multinomial_nll
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'

def getModelGivenModelOptionsAndWeightInits(args, model_params):
    # default params (can be overwritten)
    conv1_kernel_size = 21
    profile_kernel_size = 75
    num_tasks = 1  # not using multi-tasking

    filters = int(model_params['filters'])
    n_dil_layers = int(model_params['n_dil_layers'])
    counts_loss_weight = float(model_params['counts_loss_weight'])
    sequence_len = int(model_params["inputlen"])
    out_pred_len = int(model_params["outputlen"])

    print("params:")
    print("filters:" + str(filters))
    print("n_dil_layers:" + str(n_dil_layers))
    print("conv1_kernel_size:" + str(conv1_kernel_size))
    print("profile_kernel_size:" + str(profile_kernel_size))
    print("counts_loss_weight:" + str(counts_loss_weight))

    # set seeds
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)

    # input
    inp = Input(shape=(sequence_len - 1, 15), name='sequence')

    # first convolution + LeakyReLU
    x = Conv1D(filters,
               kernel_size=conv1_kernel_size,
               padding='valid',
               activation=None,
               name='bpnet_1st_conv')(inp)
    x = LeakyReLU(alpha=0.1, name='bpnet_1st_activation')(x)

    # dilated convolutions with LeakyReLU
    layer_names = [str(i) for i in range(1, n_dil_layers + 1)]
    for i in range(1, n_dil_layers + 1):
        conv_layer_name = f'bpnet_{layer_names[i-1]}conv'
        conv_x = Conv1D(filters,
                        kernel_size=3,
                        padding='valid',
                        activation=None,
                        dilation_rate=2**i,
                        name=conv_layer_name)(x)
        conv_x = LeakyReLU(alpha=0.1, name=f"bpnet_{layer_names[i-1]}_activation")(conv_x)

        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0

        x = Cropping1D((x_len - conv_x_len) // 2, name=f"bpnet_{layer_names[i-1]}crop")(x)
        x = add([conv_x, x])

    # profile prediction branch
    prof_out_precrop = Conv1D(filters=num_tasks,
                               kernel_size=profile_kernel_size,
                               padding='valid',
                               name='prof_out_precrop')(x)

    prof = tf.keras.layers.Lambda(
        lambda t: t[:, tf.shape(t)[1] // 2 - out_pred_len // 2: tf.shape(t)[1] // 2 + out_pred_len // 2],
        name="logits_profile_predictions_preflatten"
    )(prof_out_precrop)

    profile_out = Flatten(name="logits_profile_predictions")(prof)

    # counts prediction branch
    gap_combined_conv = GlobalAvgPool1D(name='gap')(x)
    count_out = Dense(num_tasks, name="logcount_predictions")(gap_combined_conv)

    model = Model(inputs=[inp], outputs=[profile_out, count_out])

    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss=[multinomial_nll, 'mse'],
                  loss_weights=[1, counts_loss_weight])

    return model

def save_model_without_bias(model, output_prefix):
    return
