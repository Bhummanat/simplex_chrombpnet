import numpy as np
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import multinomial_nll
import tensorflow as tf
import random as rn
import os

# Set a fixed random seed for reproducibility
os.environ['PYTHONHASHSEED'] = '0'

def getModelGivenModelOptionsAndWeightInits(args, model_params):
    # Default convolution parameters
    conv1_kernel_size = 21
    profile_kernel_size = 75
    num_tasks = 1  # No multitasking

    # Extract model parameters
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

    # Set random seeds
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)

    # Define input layer
    encoding_method = model_params.get("encoding_method", "one_hot")   # Defaults to one_hot

    if encoding_method == "one_hot":
        input_shape = (sequence_len, 4)
    elif encoding_method == "simplex_monomer":
        input_shape = (sequence_len, 3)
    elif encoding_method == "simplex_dimer":
        input_shape = (sequence_len - 1, 15)
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")

    inp = Input(shape=input_shape, name='sequence')

    # First convolution -> BatchNorm -> tanh
    x = Conv1D(filters,
               kernel_size=conv1_kernel_size,
               padding='valid',
               activation=None,
               name='bpnet_1st_conv')(inp)
    x = BatchNormalization(name='bpnet_1st_bn')(x)
    x = Activation("tanh", name='bpnet_1st_activation')(x)

    # Dilated convolutional blocks with residual connections
    layer_names = [str(i) for i in range(1, n_dil_layers + 1)]
    for i in range(1, n_dil_layers + 1):
        conv_layer_name = f'bpnet_{layer_names[i-1]}conv'

        # Apply dilated Conv1D -> BN -> tanh
        conv_x = Conv1D(filters,
                        kernel_size=3,
                        padding='valid',
                        activation=None,
                        dilation_rate=2**i,
                        name=conv_layer_name)(x)
        conv_x = BatchNormalization(name=f"bpnet_{layer_names[i-1]}_bn")(conv_x)
        conv_x = Activation("tanh", name=f"bpnet_{layer_names[i-1]}_activation")(conv_x)

        # Crop original x to match conv_x and add (residual connection)
        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0

        x = Cropping1D((x_len - conv_x_len) // 2, name=f"bpnet_{layer_names[i-1]}crop")(x)
        x = add([conv_x, x])

    # Profile prediction branch (conv -> crop -> flatten)
    prof_out_precrop = Conv1D(filters=num_tasks,
                              kernel_size=profile_kernel_size,
                              padding='valid',
                              name='prof_out_precrop')(x)

    prof = tf.keras.layers.Lambda(
        lambda t: t[:, tf.shape(t)[1] // 2 - out_pred_len // 2: tf.shape(t)[1] // 2 + out_pred_len // 2],
        name="logits_profile_predictions_preflatten"
    )(prof_out_precrop)

    profile_out = Flatten(name="logits_profile_predictions")(prof)

    # Count prediction branch (global avg pooling -> dense)
    gap_combined_conv = GlobalAvgPool1D(name='gap')(x)
    count_out = Dense(num_tasks, name="logcount_predictions")(gap_combined_conv)

    # Assemble model
    model = Model(inputs=[inp], outputs=[profile_out, count_out])

    # Compile with multinomial NLL and MSE losses
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss=[multinomial_nll, 'mse'],
                  loss_weights=[1, counts_loss_weight])

    return model

def save_model_without_bias(model, output_prefix):
    # This function is left empty intentionally
    return
