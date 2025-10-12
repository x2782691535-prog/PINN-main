import mne
from mne.viz.topomap import (_setup_interp, _make_head_outlines, _check_sphere, 
    _check_extrapolate)
from mne.channels.layout import _find_topomap_coords
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import (LSTM, GRU, Dense, Flatten, Bidirectional, 
    TimeDistributed, InputLayer, Activation, Reshape, concatenate, Concatenate, 
    Dropout, Conv1D, Conv2D, multiply)
from keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import pad_sequences

from scipy.optimize import minimize_scalar
# import pickle as pkl
import dill as pkl
import datetime
# from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
from time import time
from tqdm import tqdm

from . import util
from . import evaluate
from . import losses
from .custom_layers import BahdanauAttention, Attention

# Fix from: https://github.com/tensorflow/tensorflow/issues/35100
# devices = tf.config.experimental.list_physical_devices('GPU')
# if len(devices) > 0:
#     print(devices)
#     tf.config.experimental.set_memory_growth(devices, True)

class CustomPrintCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}: "
            if logs is not None:
                msg += ', '.join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))])
            print(msg)

class Net:
    ''' Class for the EsiNet neural networks.
    
    Attributes
    ----------
    fwd : mne.Forward
        The mne-python Forward model instance.
    '''
    
    def __init__(self, fwd, n_dense_layers=1, n_lstm_layers=2, 
        n_dense_units=200, n_lstm_units=32, activation_function='tanh', 
        n_filters=64, kernel_size=(3,3), l1_reg=None, n_jobs=-1, model_type='auto', 
        scale_individually=True, rescale_sources='brent', 
                 verbose=0, physics_weight=0.4, use_pinn=True, 
                 poisson_weight_initial=0.7, boundary_weight_initial=0.3,
                 dirichlet_weight_initial=0.6, robin_weight_initial=0.4):

        self._embed_fwd(fwd)
        
        self.n_dense_layers = n_dense_layers
        self.n_lstm_layers = n_lstm_layers
        self.n_dense_units = n_dense_units
        self.n_lstm_units = n_lstm_units
        self.l1_reg = l1_reg
        self.activation_function = activation_function
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        # self.default_loss = tf.keras.losses.Huber(delta=delta)
        self.default_loss = 'mean_squared_error'  # losses.weighted_huber_loss
        # self.parallel = parallel
        self.n_jobs = n_jobs
        self.model_type = model_type
        self.compiled = False
        self.scale_individually = scale_individually
        self.rescale_sources = rescale_sources
        self.verbose = verbose
        # PINN-related parameters
        self.physics_weight = physics_weight
        self.use_pinn = use_pinn
        
        # Trainable weight parameters
        self.poisson_weight_initial = poisson_weight_initial
        self.boundary_weight_initial = boundary_weight_initial
        
        # Individual boundary condition weight parameters  
        self.dirichlet_weight_initial = dirichlet_weight_initial
        self.robin_weight_initial = robin_weight_initial
        
        # Initialize trainable weights if using PINN
        if self.use_pinn:
            self._create_trainable_weights()

    def _embed_fwd(self, fwd):
        ''' Saves crucial attributes from the Forward model.
        
        Parameters
        ----------
        fwd : mne.Forward
            The forward model object.
        '''
        _, leadfield, _, _ = util.unpack_fwd(fwd)
        self.fwd = deepcopy(fwd)
        self.leadfield = leadfield
        self.n_channels = leadfield.shape[0]
        self.n_dipoles = leadfield.shape[1]
        self.interp_channel_shape = (9,9)
    
    @staticmethod
    def _handle_data_input(arguments):
        ''' Handles data input to the functions fit() and predict().
        
        Parameters
        ----------
        arguments : tuple
            The input arguments to fit and predict which contain data.
        
        Return
        ------
        eeg : mne.Epochs
            The M/EEG data.
        sources : mne.SourceEstimates/list
            The source data.

        '''
        if len(arguments) == 1:
            if isinstance(arguments[0], (mne.Epochs, mne.Evoked, mne.io.Raw, mne.EpochsArray, mne.EvokedArray, mne.epochs.EpochsFIF)):
                eeg = arguments[0]
                sources = None
            else:
                simulation = arguments[0]
                eeg = simulation.eeg_data
                sources = simulation.source_data
                # msg = f'First input should be of type simulation or Epochs, but {arguments[1]} is {type(arguments[1])}'
                # raise AttributeError(msg)

        elif len(arguments) == 2:
            eeg = arguments[0]
            sources = arguments[1]
        else:
            msg = f'Input is {type()} must be either the EEG data and Source data or the Simulation object.'
            raise AttributeError(msg)

        return eeg, sources

    def fit(self, *args, optimizer=None, learning_rate=0.001, 
        validation_split=0.05, epochs=50, metrics=None, device=None, 
        false_positive_penalty=2, delta=1., batch_size=8, loss=None, 
        sample_weight=None, return_history=False, dropout=0.2, patience=7, 
        tensorboard=False, validation_freq=1, revert_order=True):
        ''' Train the neural network using training data (eeg) and labels (sources).
        
        Parameters
        ----------
        *args : 
            Can be either two objects: 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or only one:
                simulation : esinet.simulation.Simulation
                    The Simulation object

            - two objects: EEG object (e.g. mne.Epochs) and Source object (e.g. mne.SourceEstimate)
        
        optimizer : tf.keras.optimizers
            The optimizer that for backpropagation.
        learning_rate : float
            The learning rate for training the neural network
        validation_split : float
            Proportion of data to keep as validation set.
        delta : int/float
            The delta parameter of the huber loss function
        epochs : int
            Number of epochs to train. In one epoch all training samples 
            are used once for training.
        metrics : list/str
            The metrics to be used for performance monitoring during training.
        device : str
            The device to use, e.g. a graphics card.
        false_positive_penalty : float
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses
            The loss function.
        sample_weight : numpy.ndarray
            Optional numpy array of sample weights.

        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''
        self.loss = loss
        self.dropout = dropout
    
        print("preprocess data")
        x_scaled, y_scaled = self.prep_data(args)
        
        # Early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', verbose=self.verbose, patience=patience, restore_best_weights=True)
        custom_print = CustomPrintCallback()
        
        # 动态权重更新回调

        
        if tensorboard:
            log_dir = "logs/fit/" + self.model.name + '_' + datetime.datetime.now().strftime("%m%d-%H%M")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            callbacks = [es, tensorboard_callback, custom_print]
        else:
            callbacks = [es, custom_print]
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if self.loss is None:
            def cosine_loss_func(y_true, y_pred):
                return 1 + tf.keras.losses.cosine_similarity(y_true, y_pred)
            self.loss = cosine_loss_func


        elif type(loss) == list:
            self.loss = self.loss[0](*self.loss[1])
        
        # Compile if it wasnt compiled before
        if not self.compiled:
            # Special handling for PINN models with dual outputs
            if self.model_type.lower() == 'pinn' and self.use_pinn:
                # For PINN with dual outputs, use complex physics loss functions
                def custom_main_loss(y_true, y_pred):
                    if self.loss is not None:
                        return self.loss(y_true, y_pred)
                    else:
                        return 1 + tf.keras.losses.cosine_similarity(y_true, y_pred)
                
                def custom_physics_loss(y_true, y_pred):
                    # 使用自动微分泊松方程损失
                    poisson_loss = self.autodiff_poisson_loss(y_pred)

                    # 边界条件损失（包含诺伊曼、参考电极约束）
                    boundary_loss = self.boundary_condition_loss(y_pred)

                    # 使用可训练权重组合损失
                    norm_poisson_w, norm_boundary_w, _ = self.get_normalized_trainable_weights()
                    return norm_poisson_w * poisson_loss + norm_boundary_w * boundary_loss
                
                loss_dict = {
                    'main_output': custom_main_loss,
                    'physics_output': custom_physics_loss
                }
                # 使用可训练的主损失权重
                _, _, norm_physics_w = self.get_normalized_trainable_weights()
                loss_weights = {
                    'main_output': 1.0 - norm_physics_w,
                    'physics_output': norm_physics_w
                }
                self.model.compile(optimizer, loss=loss_dict, loss_weights=loss_weights, metrics=metrics)
            else:
                # Standard single output compilation
                self.model.compile(optimizer, self.loss, metrics=metrics)
                self.compiled = True
        
        if self.model_type.lower() == 'convdip':
            # print("interpolating for convdip...")
            elec_pos = _find_topomap_coords(self.info, self.info.ch_names)
            interpolator = self.make_interpolator(elec_pos, res=self.interp_channel_shape[0])
            x_scaled_interp = deepcopy(x_scaled)
            for i, sample in enumerate(x_scaled):
                list_of_time_slices = []
                for time_slice in sample:
                    time_slice_interp = interpolator.set_values(time_slice)()[::-1]
                    time_slice_interp = time_slice_interp[:, :, np.newaxis]
                    list_of_time_slices.append(time_slice_interp)
                x_scaled_interp[i] = np.stack(list_of_time_slices, axis=0)
                x_scaled_interp[i][np.isnan(x_scaled_interp[i])] = 0
            x_scaled = x_scaled_interp
            del x_scaled_interp
            print("\t...done")
            
        print("fit model")
        n_samples = len(x_scaled)
        stop_idx = int(round(n_samples * (1-validation_split)))
        
        # Handle PINN dual output data preparation
        if self.model_type.lower() == 'pinn' and self.use_pinn:
            # For PINN, we need to prepare dual targets: source targets and reconstructed EEG targets
            def prepare_pinn_targets(x_batch, y_batch):
                # Main output targets (source estimates)
                main_targets = y_batch
                # Physics output targets (reconstructed EEG from sources)
                # Use normalized channel-averaged EEG as physics target
                physical_target = tf.reduce_mean(x_batch, axis=2)
                physical_target = tf.nn.l2_normalize(physical_target, axis=1)
                return {'main_output': main_targets, 'physics_output': physical_target}
            
            # Modified generator for PINN
            def pinn_generate_batches(x, y, batch_size, revert_order=True):
                n_batches = int(len(x) / batch_size)
                x = x[:int(n_batches*batch_size)]
                y = y[:int(n_batches*batch_size)]
                
                time_lengths = [x_let.shape[0] for x_let in x]
                idc = list(np.argsort(time_lengths).astype(int))
                
                x = [x[i] for i in idc]
                y = [y[i] for i in idc]
                while True:
                    x_pad = []
                    y_pad = []
                    for batch in range(n_batches):
                        x_batch = x[batch*batch_size:(batch+1)*batch_size]
                        y_batch = y[batch*batch_size:(batch+1)*batch_size]
                        

                        if revert_order:
                            if np.random.randn()>0:
                                x_batch = [np.flip(xx, axis=1) for xx in x_batch]
                                y_batch = [np.flip(yy, axis=1) for yy in y_batch]
                        
                        x_padlet = pad_sequences(x_batch , dtype='float32' )
                        y_padlet = pad_sequences(y_batch , dtype='float32' )
                        
                        # Prepare dual targets for PINN
                        dual_targets = prepare_pinn_targets(x_padlet, y_padlet)
                            
                        x_pad.append( x_padlet )
                        y_pad.append( dual_targets )
                    
                    new_order = np.arange(len(x_pad))
                    np.random.shuffle(new_order)
                    x_pad = [x_pad[i] for i in new_order]
                    y_pad = [y_pad[i] for i in new_order]
                    for x_padlet, y_padlet in zip(x_pad, y_pad):
                        yield (x_padlet, y_padlet)
            
            gen = pinn_generate_batches(x_scaled[:stop_idx], y_scaled[:stop_idx], batch_size, revert_order=revert_order)
            
            # Prepare validation data for PINN
            val_x = pad_sequences(x_scaled[stop_idx:], dtype='float32')
            val_y = pad_sequences(y_scaled[stop_idx:], dtype='float32')
            val_targets = prepare_pinn_targets(val_x, val_y)
            validation_data = (val_x, val_targets)
        else:
            # Standard single output training
            gen = self.generate_batches(x_scaled[:stop_idx], y_scaled[:stop_idx], batch_size, revert_order=revert_order)
            validation_data = (pad_sequences(x_scaled[stop_idx:], dtype='float32'), pad_sequences(y_scaled[stop_idx:], dtype='float32'))
        steps_per_epoch = stop_idx // batch_size

        
        if device is None:
            history = self.model.fit(x=gen, 
                    epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, verbose=0, callbacks=callbacks, 
                    sample_weight=sample_weight, validation_data=validation_data, 
                    validation_freq=validation_freq)
        else:
            with tf.device(device):
                history = self.model.fit(x=gen, 
                    epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, verbose=0, callbacks=callbacks, 
                    sample_weight=sample_weight, validation_data=validation_data, 
                    validation_freq=validation_freq)
                

        del x_scaled, y_scaled
        if return_history:
            return self, history
        else:
            return self
    @staticmethod
    def generate_batches(x, y, batch_size, revert_order=True):
            n_batches = int(len(x) / batch_size)
            x = x[:int(n_batches*batch_size)]
            y = y[:int(n_batches*batch_size)]
            
            time_lengths = [x_let.shape[0] for x_let in x]
            idc = list(np.argsort(time_lengths).astype(int))
            
            x = [x[i] for i in idc]
            y = [y[i] for i in idc]
            while True:
                x_pad = []
                y_pad = []
                for batch in range(n_batches):
                    x_batch = x[batch*batch_size:(batch+1)*batch_size]
                    y_batch = y[batch*batch_size:(batch+1)*batch_size]
                    

                    if revert_order:
                        if np.random.randn()>0:
                            # x_batch = np.flip(x_batch, axis=1)
                            # y_batch = np.flip(y_batch, axis=1)
                            x_batch = [np.flip(xx, axis=1) for xx in x_batch]
                            y_batch = [np.flip(yy, axis=1) for yy in y_batch]
                    
                    
                    
                    x_padlet = pad_sequences(x_batch , dtype='float32' )
                    y_padlet = pad_sequences(y_batch , dtype='float32' )
                    
                        
                    x_pad.append( x_padlet )
                    y_pad.append( y_padlet )
                
                new_order = np.arange(len(x_pad))
                np.random.shuffle(new_order)
                x_pad = [x_pad[i] for i in new_order]
                y_pad = [y_pad[i] for i in new_order]
                for x_padlet, y_padlet in zip(x_pad, y_pad):
                    yield (x_padlet, y_padlet)


    def prep_data(self, args):
        ''' Train the neural network using training data (eeg) and labels (sources).
        
        Parameters
        ----------
        *args : 
            Can be either two objects: 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or only one:
                simulation : esinet.simulation.Simulation
                    The Simulation object

            - two objects: EEG object (e.g. mne.Epochs) and Source object (e.g. mne.SourceEstimate)
        
        optimizer : tf.keras.optimizers
            The optimizer that for backpropagation.
        learning_rate : float
            The learning rate for training the neural network
        validation_split : float
            Proportion of data to keep as validation set.
        delta : int/float
            The delta parameter of the huber loss function
        epochs : int
            Number of epochs to train. In one epoch all training samples 
            are used once for training.
        metrics : list/str
            The metrics to be used for performance monitoring during training.
        device : str
            The device to use, e.g. a graphics card.
        false_positive_penalty : float
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses
            The loss function.
        sample_weight : numpy.ndarray
            Optional numpy array of sample weights.

        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''

        
        eeg, sources = self._handle_data_input(args)
        self.info = eeg[0].info
        self.subject = sources.subject if type(sources) == mne.SourceEstimate \
            else sources[0].subject

        # Ensure that the forward model has the same 
        # channels as the eeg object
        self._check_model(eeg)

        # Handle EEG input
        if (type(eeg) == list and isinstance(eeg[0], util.EPOCH_INSTANCES)) or isinstance(eeg, util.EPOCH_INSTANCES):
            eeg = [eeg[i].get_data(copy=True) for i, _ in enumerate(eeg)]
        else:
            eeg = [sample_eeg[0] for sample_eeg in eeg]

        for i, eeg_sample in enumerate(eeg):
            if len(eeg_sample.shape) == 1:
                eeg[i] = eeg_sample[:, np.newaxis]
            if len(eeg_sample.shape) == 3:
                eeg[i] = eeg_sample[0]
        
        # check if temporal dimension has all-equal entries
        self.equal_temporal = np.all( np.array([sample_eeg.shape[-1] for sample_eeg in eeg]) == eeg[0].shape[-1])
        
        sources = [source.data for source in sources]

        # enforce shape: list of samples, samples of shape (channels/dipoles, time)
        assert len(sources[0].shape) == 2, "sources samples must be two-dimensional"
        assert len(eeg[0].shape) == 2, "eeg samples must be two-dimensional"
        assert type(sources) == list, "sources must be a list of samples"
        assert type(eeg) == list, "eeg must be a list of samples"
        assert type(sources[0]) == np.ndarray, "sources must be a list of numpy.ndarrays"
        assert type(eeg[0]) == np.ndarray, "eeg must be a list of numpy.ndarrays"
        

        # Scale sources
        y_scaled = self.scale_source(sources)
        # Scale EEG
        x_scaled = self.scale_eeg(eeg)

        # LSTM net expects dimensions to be: (samples, time, channels)
        x_scaled = [np.swapaxes(x,0,1) for x in x_scaled]
        y_scaled = [np.swapaxes(y,0,1) for y in y_scaled]
        
        # if self.model_type.lower() == 'convdip':
        #     x_scaled = [interp(x) for x in x_scaled]

        return x_scaled, y_scaled

    def scale_eeg(self, eeg):
        ''' Scales the EEG prior to training/ predicting with the neural 
        network.

        Parameters
        ----------
        eeg : numpy.ndarray
            A 3D matrix of the EEG data (samples, channels, time_points)
        
        Return
        ------
        eeg : numpy.ndarray
            Scaled EEG
        '''
        eeg_out = deepcopy(eeg)
        
        if self.scale_individually:
            for sample, eeg_sample in enumerate(eeg):
                # Common average ref:
                for time in range(eeg_sample.shape[-1]):
                    eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
                    # eeg_out[sample][:, time] /= np.max(np.abs(eeg_sample[:, time]))
                    eeg_out[sample][:, time] /= eeg_out[sample][:, time].std()
                    
                    
        else:
            for sample, eeg_sample in enumerate(eeg):
                eeg_out[sample] = self.robust_minmax_scaler(eeg_sample)
                # Common average ref:
                for time in range(eeg_sample.shape[-1]):
                    eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
        return eeg_out
    

    def scale_source(self, source):
        ''' Scales the sources prior to training the neural network.

        Parameters
        ----------
        source : numpy.ndarray
            A 3D matrix of the source data (samples, dipoles, time_points)
        
        Return
        ------
        source : numpy.ndarray
            Scaled sources
        '''
        source_out = deepcopy(source)
        # for sample in range(source.shape[0]):
        #     for time in range(source.shape[2]):
        #         # source_out[sample, :, time] /= source_out[sample, :, time].std()
        #         source_out[sample, :, time] /= np.max(np.abs(source_out[sample, :, time]))
        for sample, _ in enumerate(source):
            # source_out[sample, :, time] /= source_out[sample, :, time].std()
            source_out[sample] /= np.max(np.abs(source_out[sample]))

        return source_out
            
    @staticmethod
    def robust_minmax_scaler(eeg):
        lower, upper = [np.percentile(eeg, 25), np.percentile(eeg, 75)]
        return (eeg-lower) / (upper-lower)

    def predict(self, eeg):
        # ===== 修正: 保证输入为float32, contiguous, 没有sub-view =====
        if hasattr(eeg, 'data'):
            data = np.ascontiguousarray(eeg.data, dtype=np.float32)
        else:
            data = np.ascontiguousarray(eeg, dtype=np.float32)
        predicted_sources = self.model.predict(data)
        # 兼容: output为1维时自动升维 (trial, voxel, time)
        if isinstance(predicted_sources, np.ndarray):
            if predicted_sources.ndim == 1:
                predicted_sources = predicted_sources[np.newaxis, :, np.newaxis]
            elif predicted_sources.ndim == 2:
                predicted_sources = predicted_sources[:, :, np.newaxis]
        predicted_sources_scaled = self._scale_p_wrap(predicted_sources, data)
        return self._postprocess(predicted_sources_scaled, eeg)

    def _scale_p_wrap(self, y_est, x_true):
        """
        兼容Pin, FC等模型输出，保障y_est始终三维 [n_trials, n_voxels, n_times]
        """
        import numpy as np
        if isinstance(y_est, np.ndarray):
            if y_est.ndim == 1:
                y_est = y_est[np.newaxis, :, np.newaxis]
            elif y_est.ndim == 2:
                y_est = y_est[:, :, np.newaxis]
        if isinstance(x_true, np.ndarray):
            if x_true.ndim == 1:
                x_true = x_true[np.newaxis, :, np.newaxis]
            elif x_true.ndim == 2:
                x_true = x_true[:, :, np.newaxis]
        try:
            for trial in range(y_est.shape[0]):
                for time in range(y_est.shape[2]):
                    scaled = self.scale_p(y_est[trial][:, time], x_true[trial][:, time])
                    y_est[trial][:, time] = scaled
        except Exception as e:
            print(f"Warning: _scale_p_wrap auto-skip: shape={y_est.shape}, error={e}")
            return y_est
        return y_est

    def _postprocess(self, predicted_sources_scaled, eeg):
        # Rescale Predicitons
        if self.rescale_sources.lower() == 'brent':
            predicted_sources_scaled = self._solve_p_wrap(predicted_sources_scaled, eeg.data)
        elif self.rescale_sources.lower() == 'rms':
            predicted_sources_scaled = self._scale_p_wrap(predicted_sources_scaled, eeg.data)
        else:
            print("Warning: <rescale_sources> is set to {self.rescale_sources}, but needs to be brent or rms. Setting to default (brent)")
            predicted_sources_scaled = self._solve_p_wrap(predicted_sources_scaled, eeg.data)



        # Convert sources (numpy.ndarrays) to mne.SourceEstimates objects
        # if verbose>0:
        #     eeg_hat = list()
        #     for predicted_source in predicted_sources_scaled:
        #         eeg_hat.append( self.leadfield @ predicted_source )
        #     residual_variances = [round(self.calc_residual_variance(M_hat, M), 2) for M_hat, M in zip(eeg_hat, eeg)]
        #     print(f"Residual Variance(s): {residual_variances} [%]")

        predicted_source_estimate = [
            util.source_to_sourceEstimate(predicted_source_scaled, self.fwd, \
                sfreq=eeg[0].info['sfreq'], tmin=eeg[0].tmin, subject=self.subject) \
                for predicted_source_scaled in predicted_sources_scaled]
        
        return predicted_source_estimate

    def calc_residual_variance(self, M_hat, M):
        return 100 *  np.sum( (M-M_hat)**2 ) / np.sum(M**2)

    def predict_sources(self, eeg):
        ''' Predict sources of 3D EEG (samples, channels, time) by reshaping 
        to speed up the process.
        
        Parameters
        ----------
        eeg : numpy.ndarray
            3D numpy array of EEG data (samples, channels, time)
        '''
        assert len(eeg[0].shape)==2, 'eeg must be a list of 2D numpy array of dim (channels, time)'
        
        # GPU优化的批量预测
        try:
            # 准备批量数据
            batch_eeg = np.stack([e[:, np.newaxis] for e in eeg], axis=0)  # (batch, time, 1, channels)
            batch_eeg = np.squeeze(batch_eeg, axis=2)  # (batch, time, channels)

            # 批量预测 - 使用更大的批次大小以充分利用GPU
            batch_size = 64
            all_predictions = []

            for i in range(0, len(eeg), batch_size):
                batch_data = batch_eeg[i:i+batch_size]

                # GPU批量预测
                if self.model_type.lower() == 'pinn' and self.use_pinn:
                    batch_outputs = self.model.predict(batch_data, verbose=0)
                    if isinstance(batch_outputs, dict):
                        batch_sources = batch_outputs['main_output'][:, 0, :].T  # (sources, batch)
                    else:
                        batch_sources = batch_outputs[:, 0, :].T
                else:
                    batch_sources = self.model.predict(batch_data, verbose=0)[:, 0, :].T

                all_predictions.append(batch_sources)

            # 合并预测结果并转换为列表格式
            if all_predictions:
                all_sources = np.concatenate(all_predictions, axis=1)  # (sources, total_samples)
                predicted_sources = [all_sources[:, i] for i in range(all_sources.shape[1])]
            else:
                predicted_sources = []

        except Exception as e:
            print(f"批量预测失败，回退到逐个预测: {e}")
            # 回退到原始方法
            if self.model_type.lower() == 'pinn' and self.use_pinn:
                model_outputs = [self.model.predict(e[:, np.newaxis], verbose=self.verbose) for e in eeg]
                predicted_sources = [output['main_output'][:,0].T for output in model_outputs]
            else:
                predicted_sources = [self.model.predict(e[:, np.newaxis], verbose=self.verbose)[:,0].T for e in eeg]

        return predicted_sources

    def predict_sources_interp(self, eeg):
        ''' Predict sources of 3D EEG (samples, channels, time) by reshaping 
        to speed up the process.
        
        Parameters
        ----------
        eeg : numpy.ndarray
            3D numpy array of EEG data (samples, channels, time)
        '''
        assert len(eeg[0].shape)==4, 'eeg must be a list of 4D numpy array of dim (time, height, width, 1)'

        predicted_sources = [self.model.predict(e[np.newaxis, :, :], verbose=self.verbose)[0] for e in eeg]
            
        # predicted_sources = np.swapaxes(predicted_sources,1,2)
        predicted_sources = [np.swapaxes(src, 0, 1) for src in predicted_sources]
        # print("shape of predicted sources: ", predicted_sources[0].shape)

        return predicted_sources

    def _solve_p_wrap(self, y_est, x_true):
        ''' Wrapper for parallel (or, alternatively, serial) scaling of 
        predicted sources.
        '''
        # assert len(y_est.shape) == 3, 'Sources must be 3-Dimensional'
        # assert len(x_true.shape) == 3, 'EEG must be 3-Dimensional'

        y_est_scaled = deepcopy(y_est)

        for trial, _ in enumerate(x_true):
            for time in range(x_true[trial].shape[-1]):
                scaled = self.solve_p(y_est[trial][:, time], x_true[trial][:, time])
                y_est_scaled[trial][:, time] = scaled

        return y_est_scaled

    # @staticmethod
    # def _prep_eeg(eeg):
    #     ''' Takes a 3D EEG array and re-references to common average and scales 
    #     individual scalp maps to max(abs(scalp_map) == 1
    #     '''
    #     assert len(eeg.shape) == 3, 'Input array <eeg> has wrong shape.'

    #     eeg_prep = deepcopy(eeg)
    #     for trial in range(eeg_prep.shape[0]):
    #         for time in range(eeg_prep.shape[2]):
    #             # Common average reference
    #             eeg_prep[trial, :, time] -= np.mean(eeg_prep[trial, :, time])
    #             # Scaling
    #             eeg_prep[trial, :, time] /= np.max(np.abs(eeg_prep[trial, :, time]))
    #     return eeg_prep

    def evaluate_mse(self, *args):
        ''' Evaluate the model regarding mean squared error
        
        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object

        Return
        ------
        mean_squared_errors : numpy.ndarray
            The mean squared error of each sample

        Example
        -------
        net = Net()
        net.fit(simulation)
        mean_squared_errors = net.evaluate_mse(simulation)
        print(mean_squared_errors.mean())
        '''

        eeg, sources = self._handle_data_input(args)
        
        y_hat = self.predict(eeg)
        
        if type(y_hat) == list:
            y_hat = np.stack([y.data for y in y_hat], axis=0)
        else:
            y_hat = y_hat.data

        if type(sources) == list:
            y = np.stack([y.data for y in sources], axis=0)
        else:
            y = sources.data
        
        if len(y_hat.shape) == 2:
            y = np.expand_dims(y, axis=0)
            y_hat = np.expand_dims(y_hat, axis=0)

        mean_squared_errors = np.mean((y_hat - y)**2, axis=1)
        return mean_squared_errors


    def evaluate_nmse(self, *args):
        ''' Evaluate the model regarding normalized mean squared error
        
        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object

        Return
        ------
        normalized_mean_squared_errors : numpy.ndarray
            The normalized mean squared error of each sample

        Example
        -------
        net = Net()
        net.fit(simulation)
        normalized_mean_squared_errors = net.evaluate_nmse(simulation)
        print(normalized_mean_squared_errors.mean())
        '''

        eeg, sources = self._handle_data_input(args)
        
        y_hat = self.predict(eeg)
        
        if type(y_hat) == list:
            y_hat = np.stack([y.data for y in y_hat], axis=0)
        else:
            y_hat = y_hat.data

        if type(sources) == list:
            y = np.stack([y.data for y in sources], axis=0)
        else:
            y = sources.data
        
        if len(y_hat.shape) == 2:
            y = np.expand_dims(y, axis=0)
            y_hat = np.expand_dims(y_hat, axis=0)

        for s in range(y_hat.shape[0]):
            for t in range(y_hat.shape[2]):
                y_hat[s, :, t] /= np.max(np.abs(y_hat[s, :, t]))
                y[s, :, t] /= np.max(np.abs(y[s, :, t]))
        
        normalized_mean_squared_errors = np.mean((y_hat - y)**2, axis=1)
        
        return normalized_mean_squared_errors

    def _build_model(self):
        ''' Build the neural network architecture using the 
        tensorflow.keras.Sequential() API. Depending on the input data this 
        function will either build:

        (1) A simple single hidden layer fully connected ANN for single time instance data
        (2) A LSTM network for spatio-temporal prediction
        (3) A PINN network for physics-informed neural network
        '''
        if self.model_type.lower() == 'convdip':
            self._build_convdip_model()
        elif self.model_type.lower() == "cnn":
            self._build_cnn_model()
        elif self.model_type.lower() == 'fc':
            self._build_fc_model()
        elif self.model_type.lower() == 'lstm':
            self._build_temporal_model()
        elif self.model_type.lower() == 'pinn':
            self._build_pinn_model()
        else:
            self._build_temporal_model()

        if self.verbose:
            self.model.summary()
    
    
    def _build_temporal_model(self):
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        name = "LSTM Model"
        self.model = keras.Sequential(name=name)
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, self.n_channels)
        
        # LSTM layers
        if isinstance(self.n_lstm_units, (tuple, list)):
            self.n_lstm_units = self.n_lstm_units[0]
        # Dropout
        if isinstance(self.dropout, (tuple, list)):
            self.dropout = self.dropout[0]

        # Model Architecture
        inputs = tf.keras.Input(shape=input_shape, name='Input')
        ## FC-Path
        fc1 = TimeDistributed(Dense(self.n_dense_units, 
                    activation=self.activation_function), 
                    name='FC1')(inputs)
        fc1 = Dropout(self.dropout)(fc1)
        direct_out = TimeDistributed(Dense(self.n_dipoles, 
            activation="linear"),
            name='FC2')(fc1)
        # LSTM Path
        lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units), dropout=self.dropout), 
            name='LSTM1')(fc1)
        mask = TimeDistributed(Dense(self.n_dipoles, 
                    activation="sigmoid"), 
                    name='Mask')(lstm1)
        
        # Combination
        multi = multiply([direct_out, mask], name="multiply")
        self.model = tf.keras.Model(inputs=inputs, outputs=multi, name='Contextualizer')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(multi))
        
    def _build_fc_model(self):
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        # self.model = keras.Sequential(name=name)
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, self.n_channels)
        # self.model.add(InputLayer(input_shape=input_shape, name='Input'))
        inputs = tf.keras.Input(shape=input_shape, name='Input_FC')
        
  
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*self.n_lstm_layers
        else:
            dropout = self.dropout
        
    
        # Hidden Dense layer(s):
        if not isinstance(self.n_dense_units, (tuple, list)):
            self.n_dense_units = [self.n_dense_units] * self.n_dense_layers
        
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*self.n_dense_layers
        else:
            dropout = self.dropout
        
        add_to = inputs
        for i in range(self.n_dense_layers):
            dense = TimeDistributed(Dense(self.n_dense_units[i], 
                activation=self.activation_function), name=f'FC_{i}')(add_to)
            dense = Dropout(dropout[i], name=f'Drop_{i}')(dense)
            add_to = dense

        # Final For-each layer:
        out = TimeDistributed(Dense(self.n_dipoles, activation='linear'), name='FC_Out')(dense)
        self.model = tf.keras.Model(inputs=inputs, outputs=out, name='FC_Model')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(out))        


        # self.model.build(input_shape=input_shape)

    def _build_cnn_model(self):
        tf.keras.backend.image_data_format() == 'channels_last'
        input_shape = (None, self.n_channels, 1)

        inputs = tf.keras.Input(shape=input_shape, name='Input_CNN')
        fc = TimeDistributed(Conv1D(self.n_filters, self.n_channels, activation=self.activation_function, name="HL_D1"))(inputs)
        fc = TimeDistributed(Flatten())(fc)
            
        # LSTM path
        lstm1 = Bidirectional(GRU(self.n_lstm_units, return_sequences=True), name='GRU')(fc)
        mask = TimeDistributed(Dense(self.n_dipoles, activation="sigmoid"), name='Mask')(lstm1)

        direct_out = TimeDistributed(Dense(self.n_dipoles, activation="tanh", name="Output_Final"))(fc)
        multi = multiply([direct_out, mask], name="multiply")

        self.model = tf.keras.Model(inputs=inputs, outputs=multi, name='Contextual_CNN_Model')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(multi))
        # model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def _build_convdip_model(self):
        # self.model = keras.Sequential(name='ConvDip-model')
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, *self.interp_channel_shape, 1)
        inputs = tf.keras.Input(shape=input_shape, name='Input_ConvDip')
        # Some definitions
              

        # Hidden Dense layer(s):
        if not isinstance(self.n_dense_units, (tuple, list)):
            self.n_dense_units = [self.n_dense_units] * self.n_dense_layers
        
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*(self.n_dense_layers+self.n_lstm_layers)
        else:
            dropout = self.dropout

        # self.model.add(InputLayer(input_shape=input_shape, name='Input'))
        add_to = inputs
        for i in range(self.n_lstm_layers):
            conv = TimeDistributed(Conv2D(self.n_filters, self.kernel_size, activation=self.activation_function, name=f"Conv2D_{i}"))(add_to)
            conv = Dropout(dropout[i], name=f'Drop_conv2d_{i}')(conv)
            add_to = conv



        flat = TimeDistributed(Flatten())(conv)
        add_to = flat
        for i in range(self.n_dense_layers):
            dense = TimeDistributed(Dense(self.n_dense_units[i], activation=self.activation_function, name=f'FC_{i}'))(add_to)
            dense = Dropout(dropout[i], name=f'Drop_FC_{i}')(dense)
            add_to = dense

        # Outout Layer
        out = TimeDistributed(Dense(self.n_dipoles, activation='linear'), name='FC_Out')(dense)
        self.model = tf.keras.Model(inputs=inputs, outputs=out, name='ConvDip_Model')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(out))
        

    @staticmethod
    def l1_sparsity(x):
        new_x = tf.math.l2_normalize(x)
        return K.mean(K.abs(new_x))
        
    def _build_pinn_model(self):
        ''' Build the Physics-Informed Neural Network (PINN) model.
        This implements the same architecture as in BCI-IV-2a左右手分类.py
        '''
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, self.n_channels)
        
        # Handle dropout
        if isinstance(self.dropout, (tuple, list)):
            dropout_rate = self.dropout[0]
        else:
            dropout_rate = self.dropout
        
        # Handle LSTM units
        if isinstance(self.n_lstm_units, (tuple, list)):
            lstm_units = self.n_lstm_units[0]
        else:
            lstm_units = self.n_lstm_units
            
        # Input layer
        inputs = tf.keras.Input(shape=input_shape, name='Input_PINN')
        
        # Deep feature extraction (similar to EEGFeatureExtractor)
        conv1 = TimeDistributed(Dense(64, activation=self.activation_function), name='PINN_Conv1')(inputs)
        conv1 = Dropout(dropout_rate, name='PINN_Conv1_Dropout')(conv1)
        
        conv2 = TimeDistributed(Dense(128, activation=self.activation_function), name='PINN_Conv2')(conv1)
        conv2 = Dropout(dropout_rate, name='PINN_Conv2_Dropout')(conv2)
        
        conv3 = TimeDistributed(Dense(256, activation=self.activation_function), name='PINN_Conv3')(conv2)
        conv3 = Dropout(dropout_rate, name='PINN_Conv3_Dropout')(conv3)
        
        # Flatten and dense layers
        deep_features = TimeDistributed(Dense(512, activation=self.activation_function), name='PINN_Deep_Features')(conv3)
        deep_features = Dropout(dropout_rate, name='PINN_Deep_Features_Dropout')(deep_features)
        
        # Source space mapping
        source_fc = TimeDistributed(Dense(self.n_dipoles, activation='linear'), name='PINN_Source_FC')(deep_features)
        
        # Physics-informed branch
        if self.use_pinn:
            # Leadfield projection for physics consistency
            def leadfield_projection(source_features):
                """Apply leadfield matrix to project from source space to electrode space"""
                leadfield_tensor = tf.constant(self.leadfield.T, dtype=tf.float32)
                projected = tf.matmul(source_features, leadfield_tensor)
                return projected
            
            # Apply leadfield projection
            projected_features = TimeDistributed(Lambda(leadfield_projection), 
                        name='PINN_Leadfield_Projection')(source_fc)
            
            # Create model with dual outputs
            self.model = tf.keras.Model(
                inputs=inputs, 
                outputs={
                    'main_output': source_fc, 
                    'physics_output': projected_features
                }, 
                name='PINN_Model'
            )
        else:
            # Single output model (no physics constraints)
            self.model = tf.keras.Model(inputs=inputs, outputs=source_fc, name='PINN_Model_NoPhysics')
        
        # Add L1 regularization if specified
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(source_fc))

    def _freeze_lstm(self):
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name or 'RNN' in layer.name:
                print(f'freezing {layer.name}')
                self.model.layers[i].trainable = False
    
    def _unfreeze_lstm(self):
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name or 'RNN' in layer.name:
                print(f'unfreezing {layer.name}')
                self.model.layers[i].trainable = True
    
    def _freeze_fc(self):
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name and not 'Out' in layer.name:
                print(f'freezing {layer.name}')
                self.model.layers[i].trainable = False

    def _unfreeze_fc(self):
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name:
                print(f'unfreezing {layer.name}')
                self.model.layers[i].trainable = True

    def _build_perceptron_model(self):
        ''' Build the artificial neural network model using Dense layers.
        '''
        input_shape = (None, None, self.n_channels)
        tf.keras.backend.set_image_data_format('channels_last')

        self.model = keras.Sequential()
        # Add hidden layers
        for _ in range(self.n_dense_layers):
            self.model.add(TimeDistributed(Dense(units=self.n_dense_units,
                                activation=self.activation_function)))
        # Add output layer
        self.model.add(TimeDistributed(Dense(self.n_dipoles, activation='linear')))
        
        # Build model with input layer
        self.model.build(input_shape=input_shape)

    


    def _check_model(self, eeg):
        ''' Check whether the current forward model has the same 
        channels as the eeg. Rebuild model if thats not the case.
        
        Parameters
        ----------
        eeg : mne.Epochs or equivalent
            The EEG instance.

        '''
        # Dont do anything if model is already built.
        if self.compiled:
            return
        
        # Else assure that channels are appropriate
        if eeg[0].ch_names != self.fwd.ch_names:
            self.fwd = self.fwd.pick_channels(eeg[0].ch_names)
            # Write all changes to the attributes
            self._embed_fwd(self.fwd)
        
        self.n_timepoints = len(eeg[0].times)
        # Finally, build model
        self._build_model()
            
    def scale_p(self, y_est, x_true):
        ''' Scale the prediction to yield same estimated GFP as true GFP

        Parameters
        ---------
        y_est : numpy.ndarray
            The estimated source vector.
        x_true : numpy.ndarray
            The original input EEG vector.
        
        Return
        ------
        y_est_scaled : numpy.ndarray
            The scaled estimated source vector.
        
        '''
        # Check if y_est is just zeros:
        if np.max(y_est) == 0:
            return y_est
        y_est = np.squeeze(np.array(y_est))
        x_true = np.squeeze(np.array(x_true))
        # Get EEG from predicted source using leadfield
        x_est = np.matmul(self.leadfield, y_est)

        gfp_true = np.std(x_true)
        gfp_est = np.std(x_est)
        scaler = gfp_true / gfp_est
        y_est_scaled = y_est * scaler
        return y_est_scaled
        
    def solve_p(self, y_est, x_true):
        '''
        Parameters
        ---------
        y_est : numpy.ndarray
            The estimated source vector.
        x_true : numpy.ndarray
            The original input EEG vector.
        
        Return
        ------
        y_scaled : numpy.ndarray
            The scaled estimated source vector.
        
        '''
        # Check if y_est is just zeros:
        if np.max(y_est) == 0:
            return y_est
        y_est = np.squeeze(np.array(y_est))
        x_true = np.squeeze(np.array(x_true))
        # Get EEG from predicted source using leadfield
        x_est = np.matmul(self.leadfield, y_est)

        # optimize forward solution
        tol = 1e-9
        options = dict(maxiter=1000, disp=False)

        # base scaling
        rms_est = np.mean(np.abs(x_est))
        rms_true = np.mean(np.abs(x_true))
        base_scaler = rms_true / rms_est

        
        opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
            bounds=(0, 1), method='bounded', options=options, tol=tol)
        
        # opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
        #     bounds=(0, 1), method='L-BFGS-B', options=options, tol=tol)

        scaler = opt.x
        y_scaled = y_est * scaler * base_scaler
        return y_scaled

    @staticmethod
    def correlation_criterion(scaler, leadfield, y_est, x_true):
        ''' Perform forward projections of a source using the leadfield.
        This is the objective function which is minimized in Net::solve_p().
        
        Parameters
        ----------
        scaler : float
            scales the source y_est
        leadfield : numpy.ndarray
            The leadfield (or sometimes called gain matrix).
        y_est : numpy.ndarray
            Estimated/predicted source.
        x_true : numpy.ndarray
            True, unscaled EEG.
        '''

        x_est = np.matmul(leadfield, y_est) 
        error = np.abs(pearsonr(x_true-x_est, x_true)[0])
        return error
    
    def save(self, path, name='model'):
        # get list of folders in path
        list_of_folders = os.listdir(path)
        model_ints = []
        for folder in list_of_folders:
            full_path = os.path.join(path, folder)
            if not os.path.isdir(full_path):
                continue
            if folder.startswith(name):
                new_integer = int(folder.split('_')[-1])
                model_ints.append(new_integer)
        if len(model_ints) == 0:
            model_name = f'\\{name}_0'
        else:
            model_name = f'\\{name}_{max(model_ints)+1}'
        new_path = path+model_name
        os.mkdir(new_path)

        # Save model only
        self.model.save(new_path)
        # self.model.save_weights(new_path)

        # copy_model = tf.keras.models.clone_model(self.model)
        # copy_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.35), loss='huber')
        # copy_model.set_weights(self.model.get_weights())


        
        # Save rest
        # Delete model since it is not serializable
        self.model = None

        with open(new_path + '\\instance.pkl', 'wb') as f:
            pkl.dump(self, f)
        
        # Attach model again now that everything is saved
        try:
            self.model = tf.keras.models.load_model(new_path, custom_objects={'loss': self.loss})
        except:
            print("Load model did not work using custom_objects. Now trying it without...")
            self.model = tf.keras.models.load_model(new_path)
        
        return self

    @staticmethod
    def make_interpolator(elec_pos, res=9, ch_type='eeg', image_interp="linear"):
        extrapolate = _check_extrapolate('auto', ch_type)
        sphere = sphere = _check_sphere(None)
        outlines = 'head'
        outlines = _make_head_outlines(sphere, elec_pos, outlines, (0., 0.))
        border = 'mean'
        extent, Xi, Yi, interpolator = _setup_interp(
            elec_pos, res, image_interp, extrapolate, outlines, border)
        interpolator.set_locations(Xi, Yi)

        return interpolator

    def autodiff_poisson_loss(self, source_activations):
        """
        使用3D自动微分计算泊松方程损失
        正确的泊松方程: ∇²φ = -∇·J/σ
        其中∇·J是电流源密度的散度
        
        参数:
            source_activations: 源空间激活，形状为(batch_size, n_sources, n_timepoints)
            
        返回:
            泊松方程损失
        """
        batch_size = tf.shape(source_activations)[0]
        n_sources = tf.shape(source_activations)[1]
        n_timepoints = tf.shape(source_activations)[2] if len(source_activations.shape) > 2 else 1
        
        # 从前向模型获取真实的源点坐标
        if hasattr(self.fwd, 'source_rr'):
            source_coords = tf.constant(self.fwd['source_rr'], dtype=tf.float32)  # (n_sources, 3)
        else:
            # 通过leadfield矩阵的结构推导源点位置
            n_dipoles_per_vertex = 3  # 通常每个顶点有3个方向的偶极子
            n_vertices = n_sources // n_dipoles_per_vertex
            
            # 创建基于大脑解剖结构的真实源点分布
            import numpy as np
            theta = tf.linspace(0.0, tf.constant(np.pi, dtype=tf.float32), int(tf.sqrt(tf.cast(n_vertices, tf.float32))))
            phi = tf.linspace(0.0, tf.constant(2.0 * np.pi, dtype=tf.float32), int(tf.sqrt(tf.cast(n_vertices, tf.float32))))
            
            theta_grid, phi_grid = tf.meshgrid(theta, phi, indexing='ij')
            theta_flat = tf.reshape(theta_grid, [-1])[:n_vertices]
            phi_flat = tf.reshape(phi_grid, [-1])[:n_vertices]
            
            # 球坐标转换为笛卡尔坐标（大脑半径约8cm）
            radius = 0.08  # 8cm in meters
            x_coords = radius * tf.sin(theta_flat) * tf.cos(phi_flat)
            y_coords = radius * tf.sin(theta_flat) * tf.sin(phi_flat)
            z_coords = radius * tf.cos(theta_flat)
            
            # 为每个顶点创建3个方向的偶极子
            source_coords_list = []
            for i in range(n_vertices):
                for j in range(n_dipoles_per_vertex):
                    source_coords_list.append([x_coords[i], y_coords[i], z_coords[i]])
            
            # 确保在GPU上处理
            with tf.device('/GPU:0'):
                # 将n_sources转换为Python整数以避免GPU-CPU切换
                if isinstance(n_sources, tf.Tensor):
                    # 在eager模式下直接获取值
                    n_sources_int = int(tf.get_static_value(n_sources) or tf.shape(source_activations)[1])
                else:
                    n_sources_int = int(n_sources)

                # 确保不超出列表长度
                n_sources_int = min(n_sources_int, len(source_coords_list))
                source_coords = tf.constant(source_coords_list[:n_sources_int], dtype=tf.float32)
        
        x_coords = source_coords[:, 0]
        y_coords = source_coords[:, 1]
        z_coords = source_coords[:, 2]
        
        poisson_losses = []
        
        # 对每个时间点计算泊松方程损失
        for t in range(n_timepoints if n_timepoints > 1 else 1):
            if n_timepoints > 1:
                activations_t = source_activations[:, :, t]  # (batch_size, n_sources)
            else:
                activations_t = tf.squeeze(source_activations, axis=-1) if len(source_activations.shape) > 2 else source_activations
            
            batch_losses = []
            
            # 对每个批次样本计算
            for b in range(batch_size):
                sample_activations = activations_t[b, :]  # (n_sources,)
                
                # 创建可微分的坐标变量
                x_var = tf.Variable(x_coords, trainable=False, dtype=tf.float32)
                y_var = tf.Variable(y_coords, trainable=False, dtype=tf.float32)
                z_var = tf.Variable(z_coords, trainable=False, dtype=tf.float32)
                
                # 使用径向基函数(RBF)插值创建连续电势场
                def rbf_interpolation(x, y, z, activations, coords):
                    """径向基函数插值"""
                    distances = tf.sqrt(tf.reduce_sum(tf.square(
                        tf.stack([x, y, z], axis=1)[:, None, :] - coords[None, :, :]
                    ), axis=2))
                    
                    # 使用高斯RBF核
                    sigma_rbf = 0.01  # RBF带宽
                    weights = tf.exp(-distances**2 / (2 * sigma_rbf**2))
                    weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
                    
                    interpolated = tf.reduce_sum(weights * activations[None, :], axis=1)
                    return interpolated
                
                # 计算电势的二阶偏导数（拉普拉斯算子）
                with tf.GradientTape() as tape_laplace:
                    with tf.GradientTape() as tape2:
                        with tf.GradientTape() as tape1:
                            tape1.watch([x_var, y_var, z_var])
                            
                            # 使用RBF插值构建连续电势场
                            phi = rbf_interpolation(x_var, y_var, z_var, sample_activations, source_coords)
                        
                        # 计算一阶偏导数
                        dphi_dx = tape1.gradient(phi, x_var)
                        dphi_dy = tape1.gradient(phi, y_var)
                        dphi_dz = tape1.gradient(phi, z_var)
                        
                        # 处理None梯度
                        if dphi_dx is None:
                            dphi_dx = tf.zeros_like(x_var)
                        if dphi_dy is None:
                            dphi_dy = tf.zeros_like(y_var)
                        if dphi_dz is None:
                            dphi_dz = tf.zeros_like(z_var)
                    
                    # 计算二阶偏导数
                    d2phi_dx2 = tape2.gradient(dphi_dx, x_var)
                    d2phi_dy2 = tape2.gradient(dphi_dy, y_var)
                    d2phi_dz2 = tape2.gradient(dphi_dz, z_var)
                
                # 处理None梯度
                if d2phi_dx2 is None:
                    d2phi_dx2 = tf.zeros_like(x_var)
                if d2phi_dy2 is None:
                    d2phi_dy2 = tf.zeros_like(y_var)
                if d2phi_dz2 is None:
                    d2phi_dz2 = tf.zeros_like(z_var)
                
                # 计算拉普拉斯算子: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
                laplacian = d2phi_dx2 + d2phi_dy2 + d2phi_dz2
                
                # 现在计算电流源密度J的散度: ∇·J
                # 在EEG中，J通常表示偶极子电流密度
                with tf.GradientTape() as tape_div:
                    tape_div.watch([x_var, y_var, z_var])
                    
                    # 将源激活解释为电流源密度的各个分量
                    # 假设每3个连续的源代表x,y,z方向的电流分量
                    n_dipoles = n_sources // 3
                    
                    # 构建连续的电流密度场
                    J_x = rbf_interpolation(x_var, y_var, z_var, 
                                          sample_activations[:n_dipoles], source_coords[:n_dipoles])
                    J_y = rbf_interpolation(x_var, y_var, z_var, 
                                          sample_activations[n_dipoles:2*n_dipoles], source_coords[n_dipoles:2*n_dipoles])
                    J_z = rbf_interpolation(x_var, y_var, z_var, 
                                          sample_activations[2*n_dipoles:3*n_dipoles], source_coords[2*n_dipoles:3*n_dipoles])
                
                # 计算电流密度的偏导数
                dJx_dx = tape_div.gradient(J_x, x_var)
                dJy_dy = tape_div.gradient(J_y, y_var) 
                dJz_dz = tape_div.gradient(J_z, z_var)
                
                # 处理None梯度
                if dJx_dx is None:
                    dJx_dx = tf.zeros_like(x_var)
                if dJy_dy is None:
                    dJy_dy = tf.zeros_like(y_var)
                if dJz_dz is None:
                    dJz_dz = tf.zeros_like(z_var)
                
                # 计算电流散度: ∇·J = ∂Jx/∂x + ∂Jy/∂y + ∂Jz/∂z
                div_J = dJx_dx + dJy_dy + dJz_dz
                
                # 正确的泊松方程: ∇²φ = +∇·J/σ (注意是正号！)
                # 从电流连续性方程推导：∇·J_total = 0
                # J_total = -σ∇φ + J_source，所以 -σ∇²φ + ∇·J_source = 0
                # 因此：∇²φ = ∇·J_source/σ (正号)
                # 重新排列为残差形式: ∇²φ - ∇·J/σ = 0
                sigma = 0.33  # 大脑灰质电导率 (S/m)
                
                # 计算泊松方程残差
                poisson_residual = laplacian - div_J / sigma
                
                # 计算损失（残差的平方）
                sample_loss = tf.reduce_mean(tf.square(poisson_residual))
                batch_losses.append(sample_loss)
            
            # 合并批次损失
            time_loss = tf.reduce_mean(tf.stack(batch_losses))
            poisson_losses.append(time_loss)
        
        # 返回所有时间点的平均损失
        return tf.reduce_mean(tf.stack(poisson_losses))

    def boundary_condition_loss(self, source_activations):
        """
        边界条件损失：实现Dirichlet、Robin边界条件
        
        参数:
            source_activations: 源空间激活，形状为(batch_size, n_sources, n_timepoints) 或 (batch_size, n_sources)
            
        返回:
            完整的边界条件损失
        """
        # 确保输入维度一致性
        if len(source_activations.shape) == 2:
            source_activations = tf.expand_dims(source_activations, axis=-1)
        
        batch_size = tf.shape(source_activations)[0]
        n_sources = tf.shape(source_activations)[1]
        n_timepoints = tf.shape(source_activations)[2]
        
        # 1. Dirichlet边界条件：φ = φ0 (边界电势固定)
        dirichlet_loss = self._dirichlet_boundary_loss(source_activations, n_sources)
        
        # 2. Robin边界条件：αφ + β∂φ/∂n = γ (混合边界条件)
        robin_loss = self._robin_boundary_loss(source_activations, n_sources)
        
        # 使用可训练权重组合边界条件损失
        (dirichlet_w, robin_w) = self.get_normalized_boundary_weights()
        
        total_boundary_loss = (
            dirichlet_w * dirichlet_loss +     # Dirichlet边界条件
            robin_w * robin_loss              # Robin边界条件
        )
        
        return total_boundary_loss
    

    
    def _dirichlet_boundary_loss(self, source_activations, n_sources):
        """
        Dirichlet边界条件：φ = φ0 (边界电势固定)
        在边界电极上施加固定的电势值约束
        """
        # 通过leadfield计算电势分布
        with tf.device('/GPU:0'):
            leadfield_tensor = tf.constant(self.leadfield, dtype=tf.float32)
            potentials = tf.einsum('bst,sc->bct', source_activations, leadfield_tensor)
        
        batch_size = tf.shape(potentials)[0]
        n_channels = tf.shape(potentials)[1] 
        n_timepoints = tf.shape(potentials)[2]
        
        # 获取边界电极索引
        boundary_electrode_indices = self._get_boundary_electrode_indices(n_channels)
        
        # 设定边界电势值（这里设为0，可以根据需要调整）
        boundary_potential_value = 0.0
        
        dirichlet_losses = []
        
        # 对每个时间点计算
        for t in range(n_timepoints):
            potentials_t = potentials[:, :, t]  # (batch, n_channels)
            batch_losses = []
            
            # 对每个批次样本计算
            for b in range(batch_size):
                sample_potentials = potentials_t[b, :]  # (n_channels,)
                
                # 获取边界电极的电势值
                boundary_potentials = tf.gather(sample_potentials, boundary_electrode_indices)
                
                # Dirichlet边界条件：φ = φ0
                # 损失 = (φ - φ0)²
                sample_loss = tf.reduce_mean(tf.square(boundary_potentials - boundary_potential_value))
                batch_losses.append(sample_loss)
            
            # 合并批次损失
            time_loss = tf.reduce_mean(tf.stack(batch_losses))
            dirichlet_losses.append(time_loss)
        
        # 返回所有时间点的平均损失
        return tf.reduce_mean(tf.stack(dirichlet_losses))
    
    def _robin_boundary_loss(self, source_activations, n_sources):
        """
        Robin边界条件：αφ + β∂φ/∂n = γ (混合边界条件)
        结合电势值和法向导数的线性组合约束
        """
        # 通过leadfield计算电势分布
        with tf.device('/GPU:0'):
            leadfield_tensor = tf.constant(self.leadfield, dtype=tf.float32)
            potentials = tf.einsum('bst,sc->bct', source_activations, leadfield_tensor)
        
        batch_size = tf.shape(potentials)[0]
        n_channels = tf.shape(potentials)[1] 
        n_timepoints = tf.shape(potentials)[2]
        
        # 获取边界电极索引
        boundary_electrode_indices = self._get_boundary_electrode_indices(n_channels)
        
        # Robin边界条件参数：αφ + β∂φ/∂n = γ
        alpha = 1.0  # 电势系数
        beta = 0.1   # 法向导数系数
        gamma = 0.0  # 目标值
        
        robin_losses = []
        
        # 对每个时间点计算
        for t in range(n_timepoints):
            potentials_t = potentials[:, :, t]  # (batch, n_channels)
            batch_losses = []
            
            # 对每个批次样本计算
            for b in range(batch_size):
                sample_potentials = potentials_t[b, :]  # (n_channels,)
                
                # 获取边界电极的电势值
                boundary_potentials = tf.gather(sample_potentials, boundary_electrode_indices)
                
                # 简化实现：假设法向导数近似为相邻电极的电势差
                # 这里使用简单的有限差分近似
                n_boundary = tf.shape(boundary_potentials)[0]
                if n_boundary > 1:
                    # 计算相邻边界电极间的电势差作为法向导数的近似
                    potential_diff = boundary_potentials[1:] - boundary_potentials[:-1]
                    normal_derivatives = tf.concat([potential_diff, [potential_diff[-1]]], axis=0)
                else:
                    normal_derivatives = tf.zeros_like(boundary_potentials)
                
                # Robin边界条件：αφ + β∂φ/∂n = γ
                robin_condition = alpha * boundary_potentials + beta * normal_derivatives
                sample_loss = tf.reduce_mean(tf.square(robin_condition - gamma))
                batch_losses.append(sample_loss)
            
            # 合并批次损失
            time_loss = tf.reduce_mean(tf.stack(batch_losses))
            robin_losses.append(time_loss)
        
        # 返回所有时间点的平均损失
        return tf.reduce_mean(tf.stack(robin_losses))
    
    def _get_standard_electrode_positions(self, n_channels):
        """
        获取标准10-20系统的电极位置
        """
        # 标准10-20系统电极位置（球面坐标转换为笛卡尔坐标）
        # 头部半径约为9.2cm
        head_radius = 0.092  # meters
        
        # 常见EEG电极的标准位置（相对于Cz的角度）
        standard_positions = [
            # Frontal electrodes
            (0.0, 0.0),      # Fz
            (-0.5, 0.3),     # F3
            (0.5, 0.3),      # F4
            (-1.0, 0.5),     # F7
            (1.0, 0.5),      # F8
            # Central electrodes  
            (0.0, 0.5),      # Cz
            (-0.5, 0.5),     # C3
            (0.5, 0.5),      # C4
            # Parietal electrodes
            (0.0, 1.0),      # Pz
            (-0.5, 0.7),     # P3
            (0.5, 0.7),      # P4
            (-1.0, 0.8),     # P7
            (1.0, 0.8),      # P8
            # Occipital electrodes
            (0.0, 1.3),      # Oz
            (-0.3, 1.1),     # O1
            (0.3, 1.1),      # O2
        ]
        
        # 将n_channels转换为Python整数
        try:
            n_channels_int = int(n_channels.numpy()) if hasattr(n_channels, 'numpy') else int(n_channels)
        except:
            n_channels_int = 22  # 默认值

        # 如果需要更多电极，均匀分布在球面上
        while len(standard_positions) < n_channels_int:
            import numpy as np
            theta_val = np.random.uniform(0, 2*np.pi)
            phi_val = np.random.uniform(0, np.pi)
            standard_positions.append((theta_val, phi_val))

        # 转换为笛卡尔坐标
        electrode_coords = []
        for i in range(n_channels_int):
            theta, phi = standard_positions[i]
            if isinstance(theta, (int, float)) and isinstance(phi, (int, float)):
                # 标准位置，直接计算
                x = head_radius * np.sin(phi) * np.cos(theta)
                y = head_radius * np.sin(phi) * np.sin(theta)
                z = head_radius * np.cos(phi)
            else:
                # TensorFlow张量，需要转换
                x = head_radius * tf.sin(phi) * tf.cos(theta)
                y = head_radius * tf.sin(phi) * tf.sin(theta)
                z = head_radius * tf.cos(phi)
            electrode_coords.append([x, y, z])
        
        return tf.constant(electrode_coords, dtype=tf.float32)
    

    

    
    def _get_boundary_electrode_indices(self, n_channels):
        """
        获取边界电极的索引
        简化实现：选择前1/4和后1/4的电极作为边界
        """
        boundary_size = n_channels // 4
        front_indices = tf.range(boundary_size, dtype=tf.int32)
        back_indices = tf.range(n_channels - boundary_size, n_channels, dtype=tf.int32)
        boundary_indices = tf.concat([front_indices, back_indices], axis=0)
        return boundary_indices
    
    def _get_electrode_adjacency_matrix(self, n_channels):
        """
        生成电极邻接矩阵
        简化实现：假设电极按顺序排列，相邻电极为邻居
        """
        # 创建简单的线性邻接关系
        adjacency = tf.zeros((n_channels, n_channels), dtype=tf.float32)
        
        # 每个电极与前后相邻的电极相连
        for i in range(n_channels):
            if i > 0:
                adjacency = tf.tensor_scatter_nd_update(
                    adjacency, 
                    [[i, i-1]], 
                    [1.0]
                )
            if i < n_channels - 1:
                adjacency = tf.tensor_scatter_nd_update(
                    adjacency, 
                    [[i, i+1]], 
                    [1.0]
                )
        
        return adjacency

    def _create_trainable_weights(self):
        """
        创建所有可训练权重参数
        """
        # 主要物理损失权重（可训练）
        self.trainable_physics_weight = tf.Variable(
            initial_value=self.physics_weight,
            trainable=True,
            dtype=tf.float32,
            name='physics_weight'
        )
        
        # 泊松方程权重（可训练）
        self.trainable_poisson_weight = tf.Variable(
            initial_value=self.poisson_weight_initial,
            trainable=True,
            dtype=tf.float32,
            name='poisson_weight'
        )
        
        # 总边界条件权重（可训练）
        self.trainable_boundary_weight = tf.Variable(
            initial_value=self.boundary_weight_initial,
            trainable=True,
            dtype=tf.float32,
            name='boundary_weight'
        )
        
        # Dirichlet边界条件权重（可训练）
        self.trainable_dirichlet_weight = tf.Variable(
            initial_value=self.dirichlet_weight_initial,
            trainable=True,
            dtype=tf.float32,
            name='dirichlet_weight'
        )
        
        # Robin边界条件权重（可训练）
        self.trainable_robin_weight = tf.Variable(
            initial_value=self.robin_weight_initial,
            trainable=True,
            dtype=tf.float32,
            name='robin_weight'
        )
    
    def get_normalized_trainable_weights(self):
        """
        获取归一化后的主要可训练权重（泊松方程 vs 边界条件）
        
        Returns
        -------
        tuple
            (normalized_poisson_weight, normalized_boundary_weight, physics_weight)
        """
        # 对物理内部权重进行归一化（泊松方程 vs 边界条件）
        physics_internal_weights = tf.nn.softmax([
            self.trainable_poisson_weight, 
            self.trainable_boundary_weight
        ])
        
        # 对主要物理权重进行Sigmoid约束到[0,1]
        normalized_physics_weight = tf.nn.sigmoid(self.trainable_physics_weight)
        
        return (
            physics_internal_weights[0],  # poisson_weight
            physics_internal_weights[1],  # boundary_weight  
            normalized_physics_weight     # physics_weight
        )
    
    def get_normalized_boundary_weights(self):
        """
        获取归一化后的边界条件权重（只包含Dirichlet和Robin边界条件）
        
        Returns
        -------
        tuple
            (dirichlet_weight, robin_weight)
        """
        # 对边界条件内部权重进行归一化（只对Dirichlet和Robin权重）
        boundary_weights = tf.nn.softmax([
            self.trainable_dirichlet_weight,
            self.trainable_robin_weight
        ])
        
        return (
            boundary_weights[0],  # dirichlet_weight
            boundary_weights[1]   # robin_weight
        )
    
    def get_all_physics_weights(self):
        """
        获取所有物理损失权重的详细信息
        
        Returns
        -------
        dict
            包含所有权重信息的字典
        """
        # 获取主要权重
        norm_poisson_w, norm_boundary_w, norm_physics_w = self.get_normalized_trainable_weights()
        
        # 获取边界条件权重
        dirichlet_w, robin_w = self.get_normalized_boundary_weights()
        
        return {
            'physics_weight': float(norm_physics_w),
            'poisson_weight': float(norm_poisson_w),
            'boundary_weight': float(norm_boundary_w),
            'dirichlet_weight': float(dirichlet_w),
            'robin_weight': float(robin_w)
        }
        
    def print_current_weights(self):
        """
        打印当前所有物理损失权重
        """
        weights = self.get_all_physics_weights()
        print("=== 当前物理损失权重 ===")
        print(f"总体物理权重: {weights['physics_weight']:.4f}")
        print(f"  └─ 泊松方程权重: {weights['poisson_weight']:.4f}")
        print(f"  └─ 边界条件总权重: {weights['boundary_weight']:.4f}")
        print(f"     ├─ Dirichlet边界: {weights['dirichlet_weight']:.4f}")
        print(f"     └─ Robin边界: {weights['robin_weight']:.4f}")
        print("========================")

class CovNet:
    ''' Class for the Covariance-based Convolutional Neural Network (CovCNN) for EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, forward, name="Cov-CNN", n_filters="auto", 
                activation_function="tanh", batch_size="auto", 
                n_timepoints=20, batch_repetitions=10,
                learning_rate=1e-3, loss="cosine_similarity",
                n_sources=10, n_orders=2, epsilon=0.5, 
                snr_range=(1,100), alpha="auto", verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        # Leadfield
        self.forward = forward
        self.leadfield = deepcopy(forward["sol"]["data"])
        self.leadfield -= self.leadfield.mean(axis=0)

        n_channels, n_dipoles = self.leadfield.shape
        if batch_size == "auto":
            batch_size = n_dipoles
        if n_filters == "auto":
            n_filters = n_channels
            
        # Store Parameters
        
        
        # Architecture
        self.name = name
        self.n_filters = n_filters
        self.activation_function = activation_function
        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        # Training Data
        self.n_timepoints = n_timepoints
        self.n_sources = n_sources
        self.n_orders = n_orders
        self.batch_repetitions = batch_repetitions
        self.snr_range = snr_range
        # Inference
        self.epsilon = epsilon
        # Other
        self.verbose = verbose
        print("Build Model:..")
        self.build_model()
        

    def predict(self, evoked) -> mne.SourceEstimate:
        source_mat = self.apply_model(evoked)
        stc = self.source_to_object(source_mat, evoked)

        return stc

    def apply_model(self, evoked) -> np.ndarray:
        y = deepcopy(evoked.data)
        y -= y.mean(axis=0)
        print("werks")
        # y /= np.linalg.norm(y, axis=0)

        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        C = y@y.T
        # Scale
        C /= abs(C).max()
        

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]
        gammas = self.model.predict(C, verbose=self.verbose)[0]
        gammas /= gammas.max()

        
        


        # Select dipole indices
        gammas[gammas<self.epsilon] = 0
        dipole_idc = np.where(gammas!=0)[0]
        print("Active dipoles: ", len(dipole_idc))

        # 1) Calculate weighted minimum norm solution at active dipoles
        n_dipoles = len(gammas)
        y = deepcopy(evoked.data)
        y -= y.mean(axis=0)
        x_hat = np.zeros((n_dipoles, n_times))
        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y

        
        return x_hat        
        
        
    def fit(self, sim, patience=7, validation_split=0.05, epochs=300, return_history=True):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),]
        
        x_train = np.stack(
            self.prep_x([ep.average().data for ep in sim.eeg_data])
            , axis=0)
        y_train = np.stack(
            self.prep_y([stc.data for stc in sim.source_data])
            , axis=0)
        # print(x_train.shape, y_train.shape)
        
        history = self.model.fit(x_train, y_train, epochs=epochs, 
            validation_split=validation_split, callbacks=callbacks)

        if return_history:
            return self, history
        return self
    def prep_y(self, y):
        n_samples = len(y)
        y_scaled = []

        for i in range(n_samples):
            y_sample = y[i]

            y_sample = np.mean(abs(y_sample), axis=1)
            thr = y_sample.max()*1e-3
            y_sample = (y_sample>thr).astype(float)
            y_scaled.append(y_sample)
        return y_scaled


    def prep_x(self, x):
        n_samples = len(x)
        C_scaled = []
        for i in range(n_samples):
            x_sample = x[i]
            # Common Average Reference
            x_sample -= x_sample.mean(axis=0)
            x_sample /= np.linalg.norm(x_sample, axis=0)
            C = x_sample @ x_sample.T
            C /= abs(C).max()
            C_scaled.append(C)
        return C_scaled
            

    def build_model(self,):
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(n_channels, n_channels, 1), name='Input')

        cnn1 = Conv2D(self.n_filters, (1, n_channels),
                    activation=self.activation_function, padding="valid",
                    name='CNN1')(inputs)

        flat = Flatten()(cnn1)
        
        fc1 = Dense(200, 
            activation=self.activation_function, 
            name='FC1')(flat)
        out = Dense(n_dipoles, 
            activation="sigmoid", 
            name='Output')(fc1)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CovCNN')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model
 
    def source_to_object(self, source_mat, evoked):
        ''' Converts the source_mat matrix to an mne.SourceEstimate object '''
        # Convert source to mne.SourceEstimate object
        source_model = self.forward['src']
        vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
        tmin = evoked.tmin
        sfreq = evoked.info["sfreq"]
        tstep = 1/sfreq
        subject = evoked.info["subject_info"]

        if type(subject) == dict:
            subject = "bst_raw"

        if subject is None:
            subject = "fsaverage"
        
        stc = mne.SourceEstimate(source_mat, vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=self.verbose)
        return stc

    def save(self, path, name='model'):
        # get list of folders in path
        list_of_folders = os.listdir(path)
        model_ints = []

        for folder in list_of_folders:
            full_path = os.path.join(path, folder)
            if not os.path.isdir(full_path):
                continue
            if folder.startswith(name):
                new_integer = int(folder.split('_')[-1])
                model_ints.append(new_integer)
        if len(model_ints) == 0:
            model_name = f'\\{name}_0'
        else:
            model_name = f'\\{name}_{max(model_ints)+1}'

        new_path = path+model_name
        os.mkdir(new_path)

        # Save model only
        self.model.save(new_path)

        
        # Save rest
        # Delete model since it is not serializable
        self.model = None

        with open(new_path + '\\instance.pkl', 'wb') as f:
            pkl.dump(self, f)
        
        # Attach model again now that everything is saved
        try:
            self.model = tf.keras.models.load_model(new_path, custom_objects={'loss': self.loss})
        except:
            print("Load model did not work using custom_objects. Now trying it without...")
            self.model = tf.keras.models.load_model(new_path)
        
        return self

def build_nas_lstm(hp):
    ''' Find optimal model using keras tuner.
    '''
    n_dipoles = 1284
    n_channels = 61
    n_lstm_layers = hp.Int("lstm_layers", min_value=0, max_value=3, step=1)
    n_dense_layers = hp.Int("dense_layers", min_value=0, max_value=3, step=1)
    activation_out = 'linear'  # hp.Choice(f"activation_out", ["tanh", 'sigmoid', 'linear'])
    activation = 'relu'  # hp.Choice('actvation_all', all_acts)

    model = keras.Sequential(name='LSTM_NAS')
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = (None, n_channels)
    model.add(InputLayer(input_shape=input_shape, name='Input'))

    # LSTM layers
    for i in range(n_lstm_layers):
        n_lstm_units = hp.Int(f"lstm_units_l-{i}", min_value=25, max_value=500, step=1)
        dropout = hp.Float(f"dropout_lstm_l-{i}", min_value=0, max_value=0.5)
        model.add(Bidirectional(LSTM(n_lstm_units, 
            return_sequences=True, input_shape=input_shape, 
            dropout=dropout, activation=activation), 
            name=f'LSTM{i}'))
    # Hidden Dense layer(s):
    for i in range(n_dense_layers):
        n_dense_units = hp.Int(f"dense_units_l-{i}", min_value=50, max_value=1000, step=1)
        dropout = hp.Float(f"dropout_dense_l-{i}", min_value=0, max_value=0.5)

        model.add(TimeDistributed(Dense(n_dense_units, 
            activation=activation), name=f'FC_{i}'))
        model.add(Dropout(dropout, name=f'DropoutLayer_dense_{i}'))

    # Final For-each layer:
    model.add(TimeDistributed(
        Dense(n_dipoles, activation=activation_out), name='FC_Out')
    )
    model.build(input_shape=input_shape)
    momentum = hp.Float('Momentum', min_value=0, max_value=0.9)
    nesterov = hp.Choice('Nesterov', [False, True])
    learning_rate = hp.Choice('learning_rate', [0.01, 0.001])
    optimizer = hp.Choice("Optimizer", [0,1,2])
    optimizers = [keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum), keras.optimizers.Adam(learning_rate=learning_rate), keras.optimizers.SGD(learning_rate=learning_rate, nesterov=nesterov)]
    model.compile(
        optimizer=optimizers[optimizer],
        loss="huber",
        # metrics=[tf.keras.metrics.AUC()],
        # metrics=[evaluate.modified_auc_metric()],
        metrics=[evaluate.auc],
    )
    return model

    