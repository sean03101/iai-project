import os
import shap
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl

import traceback

def calculate_and_visualize_shap_imvlstm(model, X_train_scaled, feature_names, device, num_background=100, num_test_samples=100):
    """
    This function wraps a given model for use with SHAP DeepExplainer, generates SHAP values for a subset of test data,
    and visualizes the average impact of each feature on the model's predictions. The function is specifically designed
    for models that return multiple outputs, such as those with multiple layers or heads, by focusing on the primary output.

    Parameters:
    - model: The PyTorch model to be explained, which is assumed to return multiple outputs.
    - X_train_scaled: Scaled training data used to select a background distribution for SHAP values computation.
    - feature_names: A list of names for each feature in the dataset.
    - device: The PyTorch device (e.g., 'cuda' or 'cpu') to which the model and data should be sent.
    - num_background: The number of background samples to use when initializing the SHAP DeepExplainer.
    - num_test_samples: The number of test samples to explain.

    Steps:
    1. Wraps the original model to ensure it returns a single output for SHAP compatibility.
    2. Samples background and test data from the provided dataset.
    3. Initializes a SHAP DeepExplainer with the model wrapper and background data.
    4. Computes SHAP values for the test data, handling models with multiple outputs.
    5. Averages the SHAP values over time (if applicable) to simplify interpretation.
    6. Visualizes the average impact of each feature on the model's predictions using SHAP's summary plot.

    Returns:
    - shap_values: The computed SHAP values for the test data.
    - explainer: The SHAP DeepExplainer instance used for computations.

    This function is particularly useful for interpreting complex time-series models like IMV-LSTM, providing insights
    into how different features contribute to the model's predictions on average.
    """
    class ModelWrapper(torch.nn.Module):
        # Creating a pipeline that returns only the prediction values due to multiple outputs from the model.
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            output, _, __ = self.model(x)
            return output
    
    model_wrapper = ModelWrapper(model).to(device)

    background_data = torch.tensor(random.sample(X_train_scaled, num_background), dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(model_wrapper, background_data)

    all_test_data = torch.tensor(random.sample(X_train_scaled, num_test_samples), dtype=torch.float32).to(device)
    
    model.eval()
    torch.backends.cudnn.enabled = False
    shap_values = explainer.shap_values(all_test_data, check_additivity=False)
    torch.backends.cudnn.enabled = True
    
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])
    
    average_shap_values_over_time = np.mean(shap_values, axis=1)
    shap_values_averaged = average_shap_values_over_time.reshape(shap_values.shape[0], -1)
    all_test_data_averaged = np.mean(all_test_data.cpu().numpy(), axis=1)
    
    feature_names = feature_names
    shap.summary_plot(shap_values_averaged, all_test_data_averaged, feature_names=feature_names)

    return shap_values, explainer


def calculate_and_visualize_shap_lstm(model, X_train_scaled, feature_names, device, num_background=100, num_test_samples=100):
    """
    Wraps an LSTM model for SHAP value computation and visualization, specifically handling models that may return multiple outputs. 
    It calculates SHAP values for a selected subset of test data to explain the model's predictions and visualizes the average 
    importance of each feature across all input timesteps.

    Parameters:
    - model: The LSTM model to explain, which may return a single output or a tuple of outputs.
    - X_train_scaled: The scaled training data, from which background and test samples will be randomly selected.
    - feature_names: A list of feature names for the input data, used for labeling in the visualization.
    - device: The computation device ('cuda' or 'cpu') for PyTorch operations.
    - num_background: The number of samples from the training data to use as background for the SHAP DeepExplainer.
    - num_test_samples: The number of samples from the training data to explain with SHAP values.

    The function:
    1. Wraps the provided model in a custom PyTorch module that ensures compatibility with SHAP's DeepExplainer by returning a consistent single output format.
    2. Prepares background and test datasets from the provided scaled training data.
    3. Initializes a SHAP DeepExplainer with the wrapped model and background dataset.
    4. Calculates SHAP values for the test dataset, accommodating models that return multiple outputs.
    5. Averages the SHAP values across all timesteps to simplify interpretation.
    6. Visualizes the average SHAP values for each feature across the selected test samples using a summary plot.

    Returns:
    - shap_values: An array of SHAP values for the test samples.
    - explainer: The SHAP DeepExplainer used for computing SHAP values.

    This function is wrapped in a try-except block to handle potential errors gracefully and print exceptions, making debugging easier.
    It is designed to offer insights into the feature importance of LSTM models in time-series predictions.
    """    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return outputs
    
    model_wrapper = ModelWrapper(model).to(device)

    background_data = torch.tensor(random.sample(X_train_scaled, num_background), dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(model_wrapper, background_data)
    all_test_data = torch.tensor(random.sample(X_train_scaled, num_test_samples), dtype=torch.float32).to(device)
    
    model.eval()
    torch.backends.cudnn.enabled = False
    shap_values = explainer.shap_values(all_test_data, check_additivity=False)
    torch.backends.cudnn.enabled = True
    
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])
    
    average_shap_values_over_time = np.mean(shap_values, axis=1)
    shap_values_averaged = average_shap_values_over_time.reshape(shap_values.shape[0], -1)
    all_test_data_averaged = np.mean(all_test_data.cpu().numpy(), axis=1)
    
    feature_names = feature_names
    shap.summary_plot(shap_values_averaged, all_test_data_averaged, feature_names=feature_names)
        
    return shap_values, explainer

def calculate_and_visualize_shap_cnn(model, X_train_tensor, X_test_tensor, feature_names, device):
    """
    This function wraps a Convolutional Neural Network (CNN) model to compute SHAP values and visualize the
    feature importance for a subset of test data. It is designed for models dealing with high-dimensional data,
    such as images or time-series data represented in tensor format.

    Parameters:
    - model: The CNN model to be explained.
    - X_train_tensor: Tensor of training data, used to select a background distribution for SHAP explanations.
    - X_test_tensor: Tensor of test data, from which a subset will be explained.
    - feature_names: A list of names for each feature in the dataset, used for labeling in the visualization.
    - device: The computation device (e.g., 'cuda' or 'cpu') to use for model and data operations.

    The process:
    1. Wraps the provided CNN model in a custom PyTorch module to ensure compatibility with SHAP's DeepExplainer.
    2. Selects a random subset of the training data to serve as background for the SHAP explanations.
    3. Initializes a SHAP DeepExplainer with the wrapped model and the selected background data.
    4. Computes SHAP values for a randomly selected subset of the test data.
    5. Averages the SHAP values across the input dimension to simplify the visualization of feature importance.
    6. Uses SHAP's summary plot to visualize the average feature importance across the selected test samples.

    Returns:
    - shap_values: An array of SHAP values for the explained test samples, providing insight into how each feature
      contributes to the model's output.
    - explainer: The SHAP DeepExplainer instance used for computing the SHAP values.

    This function allows for the interpretation of CNN models by highlighting the importance of each input feature
    (e.g., pixels in an image) in the model's predictions. It is particularly useful for understanding complex
    high-dimensional data models.
    """
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            output = self.model(x)
            return output
    
    model_wrapper = ModelWrapper(model).to(device)
    
    background_data = X_train_tensor[torch.randperm(len(X_train_tensor))[:500]].to(device)
    explainer = shap.DeepExplainer(model_wrapper, background_data)

    all_test_data = X_test_tensor[torch.randperm(len(X_test_tensor))[:500]].to(device)

    model.eval()
    torch.backends.cudnn.enabled = False

    shap_values = explainer.shap_values(all_test_data,  check_additivity=False)

    torch.backends.cudnn.enabled = True

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])

    average_shap_values_over_time = np.mean(shap_values, axis=1)
    shap_values_averaged = average_shap_values_over_time.reshape(shap_values.shape[0], -1)
    all_test_data_averaged = np.mean(all_test_data.cpu().numpy(), axis=1)

    shap.summary_plot(shap_values_averaged, all_test_data_averaged, feature_names=feature_names) 

    return shap_values, explainer

def feature_importance(shap_values, feature_names, num, save_path):

    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0,num))

    feature_names = feature_names

    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap_values
    })

    top_feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    top_feature_importances.plot(kind='bar', x='feature', y='importance', legend=False)
    plt.title('Top 10 Feature Importances using SHAP')
    plt.ylabel('Average Impact on Model Output')
    plt.xticks(rotation=45)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,j))
for j in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,j))

def shap_image_plot(shap_values,
          pixel_values = None,
          labels = None,
          true_labels = None,
          width = 20,
          aspect = 0.2,
          hspace = 0.2,
          labelpad = None,
          features = None,
          show = True):
    """Plots SHAP values for image inputs.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shape
        (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being
        explained.

    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image.
        It should be the same
        shape as each array in the ``shap_values`` list of arrays.

    labels : list or np.ndarray
        List or ``np.ndarray`` (# samples x top_k classes) of names for each of the
        model outputs that are being explained.

    true_labels: list
        List of a true image labels to plot.

    width : float
        The width of the produced matplotlib plot.

    labelpad : float
        How much padding to use around the model output labels.

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    Examples
    --------

    See `image plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/image.html>`_.

    """

    colors = []
    for j in np.linspace(1, 0, 100):
        colors.append((30./255, 136./255, 229./255,j))
    for j in np.linspace(0, 1, 100):
        colors.append((255./255, 13./255, 87./255,j))
    cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)
    
    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        if len(shap_exp.output_dims) == 1:
            shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]
        elif len(shap_exp.output_dims) == 0:
            shap_values = shap_exp.values
        else:
            raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")
        if pixel_values is None:
            pixel_values = shap_exp.data
        if labels is None:
            labels = shap_exp.output_names

    # multi_output = True
    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    # labels: (rows (images) x columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    x = pixel_values
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = pl.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=(15,20))
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure we have a 2D array for grayscale
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                    0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
            x_curr_disp = x_curr
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)

            # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
            flat_vals = x_curr.reshape([x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]).T
            flat_vals = (flat_vals.T - flat_vals.mean(1)).T
            means = kmeans(flat_vals, 3, round_values=False).data.T.reshape([x_curr.shape[0], x_curr.shape[1], 3])
            x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                    np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1)))
            x_curr_disp[x_curr_disp > 1] = 1
            x_curr_disp[x_curr_disp < 0] = 0
        else:
            x_curr_gray = x_curr
            x_curr_disp = x_curr

        axes[row, 0].imshow(x_curr_disp, cmap=pl.get_cmap('gray'))
        if true_labels:
            axes[row, 0].set_title(true_labels[row], **label_kwargs)
        axes[row, 0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(len(shap_values)):
            axes[row, i + 1].set_yticks(range(len(features)))
            axes[row, i + 1].set_yticklabels(features, rotation=0)
            if labels is not None:
                axes[row, i + 1].set_title(labels[row, i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            
            axes[row, i + 1].xaxis.set_visible(False)   
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                      aspect=fig_size[0] / aspect)
    cb.outline.set_visible(False)
    for ax in axes[:, 0]:
        ax.remove()  
    if show:
        pl.show()

def cnn_regression_outlier_visualization(outlier_elements, X_test_scaled, explainer, feature_names, all_predictions, all_predictions_class, all_labels_class, device, save_path='1D_CNN_model/outlier'):
    """
    Visualizes outliers identified in 1D CNN regression model predictions using SHAP values. It generates decision plots,
    force plots, and image plots for each outlier sample to understand the contribution of each feature to the prediction anomaly.

    Parameters:
    - outlier_elements: A list of indices for the outlier samples in the test dataset.
    - X_test_scaled: The scaled test dataset used for model predictions.
    - explainer: A SHAP DeepExplainer object initialized with the model and a background dataset.
    - feature_names: A list of feature names corresponding to the input features of the model.
    - all_predictions: A list or array of all prediction values from the model.
    - all_predictions_class: A list or array of classified prediction results (e.g., normal or abnormal).
    - all_labels_class: A list or array of the true classified labels for the test dataset.
    - device: The computation device ('cuda' or 'cpu') for PyTorch operations.
    - save_path: The directory path where the generated visualization plots will be saved.

    The function:
    1. Converts the outlier samples from the test dataset to tensors and adjusts their dimensions for the model input.
    2. Computes SHAP values for these outlier samples to analyze feature contributions.
    3. Generates and saves three types of SHAP visualizations for each outlier sample:
       - Decision plots: Show the cumulative impact of features leading to the model's prediction.
       - Force plots: Illustrate how each feature contributes to moving the model output from the base value.
       - Image plots: Display the feature contributions in an image format for each sample.
    4. Creates the specified save directory if it doesn't exist.
    5. Outputs the prediction details (predicted value, predicted label, and true label) for each outlier sample.

    This visualization assists in diagnosing why certain samples were identified as outliers by the model, offering
    insights into potential reasons for anomalous predictions based on feature contributions.
    """
    outlier_elements_array = np.array(outlier_elements)
    outlier_sample = [X_test_scaled[i] for i in outlier_elements_array]
    outlier_sample_tensor = torch.tensor(outlier_sample, dtype=torch.float32).to(device)
    outlier_sample_tensor = outlier_sample_tensor.permute(0, 2, 1)

    shap_values = explainer.shap_values(outlier_sample_tensor, check_additivity=False)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # SHAP Decision Plot, Force Plot, Image Plot
    for i in range(len(outlier_elements)):
        shap.decision_plot(explainer.expected_value, np.mean(shap_values[i], axis=1), features=feature_names, show=False)
        plt.savefig(f"{save_path}/decision_plot_sample{outlier_elements[i]}.png", dpi=300, bbox_inches='tight')
        plt.clf()

        shap.force_plot(explainer.expected_value, np.sum(shap_values[i], axis=1), features=feature_names, matplotlib=True, show=False)
        plt.savefig(f"{save_path}/force_plot_{outlier_elements[i]}.png", dpi=300, bbox_inches='tight')
        plt.clf()

        shap_image_plot(np.expand_dims(shap_values[i], axis=-1), -np.expand_dims(np.array(X_test_scaled[0]), -1), show=False, width=100, aspect=0.05, hspace=0.05 , features=feature_names)
        plt.savefig(f"{save_path}/imageplot_{outlier_elements[i]}.png",dpi=300, bbox_inches='tight')
        plt.clf()

    for idx in outlier_elements:
        predicted_value = all_predictions[idx]
        predicted_label = all_predictions_class[idx]
        true_label = all_labels_class[idx]
        print(f"Index: {idx}, Predicted Value: {predicted_value}, Predicted Label: {predicted_label}, True Label: {true_label}")

def cnn_classification_outlier_visualization(outlier_elements, X_test_scaled, explainer, all_predictions, all_labels, feature_names, device, save_path='1D_CNN_model_clf/outlier'):
    """
    Visualizes outliers identified in 1D CNN classification model predictions using SHAP values. This function generates
    decision plots, force plots, and image plots for each outlier sample to dissect the influence of individual features
    on the anomalous classification outcome.

    Parameters:
    - outlier_elements: Indices of the outlier samples in the test dataset.
    - X_test_scaled: Scaled test dataset used for model predictions.
    - explainer: A SHAP DeepExplainer object initialized with the model and a background dataset for computing SHAP values.
    - all_predictions: Prediction outcomes from the model for the test dataset.
    - all_labels: Actual labels for the test dataset.
    - feature_names: Names of the features in the dataset, used for visualization labeling.
    - device: The computation device ('cuda' or 'cpu') for PyTorch operations.
    - save_path: Directory path where the generated visualization plots will be stored.

    Workflow:
    1. Converts the selected outlier samples and their labels from the test dataset into tensors suitable for the model.
    2. Computes SHAP values for these outlier samples to identify feature contributions to the prediction.
    3. Generates and saves SHAP visualizations for each outlier sample, including:
       - Decision plots showing the cumulative impact of features on the model's output.
       - Force plots illustrating the positive and negative contributions of features to the model's prediction.
       - Image plots visualizing the feature contributions for a more intuitive understanding.
    4. Creates the output directory if it does not exist.
    5. Prints the prediction details (predicted value and true label) for each outlier sample.

    This function facilitates a deeper understanding of the CNN model's classification decisions, especially for outliers,
    by highlighting how different features influenced the model's predictions.
    """
    outlier_elements_array = np.array(outlier_elements)
    outlier_sample = [X_test_scaled[i] for i in outlier_elements_array]
    outlier_prediction = [int(all_predictions[i]) for i in outlier_elements_array]

    outlier_sample_tensor = torch.tensor(outlier_sample, dtype=torch.float32).to(device)
    outlier_sample_tensor = outlier_sample_tensor.permute(0, 2, 1)
    shap_values = explainer.shap_values(outlier_sample_tensor, check_additivity=False)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for sample_num in range(len(outlier_elements)):
        shap.decision_plot(explainer.expected_value[outlier_prediction[sample_num]],np.sum(shap_values[outlier_prediction[sample_num]][sample_num], axis=1), features=feature_names, show=False)
        plt.savefig(f"{save_path}/decision_plot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()

        shap.force_plot(explainer.expected_value[outlier_prediction[sample_num]], np.sum(shap_values[outlier_prediction[sample_num]][sample_num], axis=1), features=feature_names, matplotlib=True, show=False)
        plt.savefig(f"{save_path}/force_plot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()
        
        shap_image_plot(np.expand_dims(shap_values[outlier_prediction[sample_num]][sample_num], axis=-1),-np.expand_dims(np.array(X_test_scaled[0]), -1), show=False, width=100, aspect=0.05, hspace=0.05 , features=feature_names)
        plt.savefig(f"{save_path}/imageplot_{outlier_elements[sample_num]}.png",dpi=300, bbox_inches='tight')
        plt.clf()

    for idx in outlier_elements:
        predicted_value = all_predictions[idx]
        true_label = all_labels[idx]
        print(f"Index: {idx}, Predicted Value: {predicted_value}, True Label: {true_label}")


def lstm_regression_outlier_visualization(outlier_elements, model,explainer, X_test_scaled, feature_names, all_predictions, all_labels_class, device, save_path='LSTM/outlier'):
    """
    Visualizes outliers in LSTM regression model predictions using SHAP values. This function generates decision plots,
    force plots, and image plots for each identified outlier sample to explore the impact of each feature on the prediction
    deviation.

    Parameters:
    - outlier_elements: Indices of the outlier samples in the test dataset.
    - model: The LSTM model used for regression predictions.
    - explainer: A SHAP DeepExplainer object initialized with the model and a background dataset for computing SHAP values.
    - X_test_scaled: Scaled test dataset used for model predictions.
    - feature_names: Names of the features in the dataset, used for visualization labeling.
    - all_predictions: Prediction outcomes from the model for the test dataset.
    - all_labels_class: Classified labels for the test dataset indicating the true outcome.
    - device: The computation device ('cuda' or 'cpu') for PyTorch operations.
    - save_path: Directory path where the generated visualization plots will be stored.

    Workflow:
    1. Prepares the outlier samples from the test dataset for the LSTM model input.
    2. Computes SHAP values for these outlier samples to examine feature contributions to the prediction anomalies.
    3. Generates and saves SHAP visualizations for each outlier sample, including:
       - Decision plots to show the cumulative impact of features on the model's prediction.
       - Force plots to illustrate the overall contribution of features towards moving the model output from the base value.
       - Image plots for a visual representation of feature contributions in a more intuitive format.
    4. Ensures the output directory is created if it does not exist.
    5. Outputs the prediction details (predicted value, predicted and true labels) for each outlier sample.

    This function aids in the interpretation of LSTM regression models by identifying and visualizing the reasons behind
    outlier predictions, thereby providing insights into model behavior and feature importance.
    """
    outlier_elements_array = np.array(outlier_elements)
    outlier_sample = [X_test_scaled[i] for i in outlier_elements_array]

    model.train()
    shap_values = explainer.shap_values(torch.tensor(outlier_sample, dtype=torch.float32).to(device), check_additivity=False)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for sample_num, idx in enumerate(outlier_elements):
        shap.decision_plot(explainer.expected_value, shap_values[sample_num], features=feature_names, show=False)
        plt.savefig(f"{save_path}/decision_plot_{idx}.png", dpi=300, bbox_inches='tight')
        plt.clf()
    
        shap.force_plot(explainer.expected_value, np.sum(shap_values[sample_num], axis=0), features=feature_names, matplotlib=True, show=False)
        plt.savefig(f"{save_path}/force_plot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()
    
        shap_image_plot(np.expand_dims(shap_values[sample_num].transpose(), axis=-1), -np.expand_dims(np.array(outlier_sample[0]).transpose(), -1), show=False, width=100, aspect=0.05, hspace=0.05 , features=feature_names)
        plt.savefig(f"{save_path}/imageplot_{outlier_elements[sample_num]}.png",dpi=300, bbox_inches='tight')
        plt.clf()
    
    for idx in outlier_elements:
        predicted_value = all_predictions[idx]
        predicted_label = all_labels_class[idx]
        true_label = all_labels_class[idx]
        print(f"Index: {idx}, Predicted Value: {predicted_value}, Predicted Label: {predicted_label}, True Label: {true_label}")


def lstm_classification_outlier_visualization(outlier_elements, model,explainer, X_test_scaled,feature_names, all_predictions, all_labels, device, save_path='LSTM_clf/outlier'):
    """
    Visualizes outliers in LSTM classification model predictions using SHAP values. It creates decision plots, force plots,
    and image plots for each outlier sample to dissect the influence of individual features on the anomalous classification.

    Parameters:
    - outlier_elements: Indices of the outlier samples in the test dataset.
    - model: The LSTM model used for classification predictions.
    - explainer: A SHAP DeepExplainer object initialized with the model and a background dataset for SHAP value computation.
    - X_test_scaled: Scaled test dataset used for model predictions.
    - feature_names: Names of the features in the dataset, used for visualization labeling.
    - all_predictions: Prediction outcomes from the model for the test dataset.
    - all_labels: Actual labels for the test dataset.
    - device: The computation device ('cuda' or 'cpu') for PyTorch operations.
    - save_path: Directory path where the generated visualization plots will be stored.

    Workflow:
    1. Prepares the outlier samples from the test dataset for LSTM model input.
    2. Computes SHAP values for these outlier samples to understand feature contributions to the misclassification.
    3. Generates and saves SHAP visualizations for each outlier sample, including:
       - Decision plots to show the cumulative impact of features on the model's classification output.
       - Force plots to illustrate the positive and negative contributions of features to the classification decision.
       - Image plots for a visual representation of feature contributions in an intuitive format.
    4. Creates the output directory if it does not exist.
    5. Prints the prediction details (predicted value and true label) for each outlier sample.

    This function helps in understanding the reasons behind specific classification decisions by LSTM models,
    especially for outlier predictions, by highlighting how different features influenced the model's predictions.
    """
    outlier_elements_array = np.array(outlier_elements)
    outlier_sample = [X_test_scaled[i] for i in outlier_elements_array]
    outlier_prediction = [int(all_predictions[i]) for i in outlier_elements_array]
    
    # Set the model to training mode and calculate SHAP values
    model.train()
    shap_values = explainer.shap_values(torch.tensor(outlier_sample, dtype=torch.float32).to(device), check_additivity=False)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # decision plots
    for sample_num in range(len(outlier_elements)):
        shap.decision_plot(explainer.expected_value[outlier_prediction[sample_num]],
                           np.sum(shap_values[outlier_prediction[sample_num]][sample_num], axis=0),
                           features=feature_names, show=False)
        plt.savefig(f"{save_path}/decision_plot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()

    # force plots
        shap.force_plot(explainer.expected_value[outlier_prediction[sample_num]],
                        np.sum(shap_values[outlier_prediction[sample_num]][sample_num], axis=0),
                        features=feature_names, matplotlib=True, show=False)
        plt.savefig(f"{save_path}/force_plot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()

    # the image plots
        shap_image_plot(np.expand_dims(shap_values[outlier_prediction[sample_num]][sample_num].transpose(), axis=-1),
                        -np.expand_dims(np.array(X_test_scaled[outlier_elements[sample_num]]).transpose(), -1),
                        show=False, width=100, aspect=0.05, hspace=0.05, features=feature_names)
        plt.savefig(f"{save_path}/imageplot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()

    # Print predictions and true labels for outliers
    for idx in outlier_elements:
        predicted_value = all_predictions[idx]
        true_label = all_labels[idx]
        print(f"Index: {idx}, Predicted Value: {predicted_value}, True Label: {true_label}")


def imvlstm_regression_outlier_visualization(outlier_elements, model,explainer, X_test_scaled,X_train_scaled,X_train,feature_names,all_predictions, all_predictions_class, all_labels_class, device, save_path='IMV_LSTM/outlier'):
    """
    Visualizes outliers in IMV-LSTM regression model predictions using SHAP values and model-specific outputs (alphas and betas).
    It creates various plots to explore the impact of individual features and timesteps on the anomalous predictions.

    Parameters:
    - outlier_elements: Indices of the outlier samples in the test dataset.
    - model: The IMV-LSTM model used for regression predictions.
    - explainer: A SHAP DeepExplainer object initialized with the model and a background dataset for SHAP value computation.
    - X_test_scaled: Scaled test dataset used for model predictions.
    - X_train_scaled, X_train: Scaled and original training datasets, used for feature and timestep visualization.
    - feature_names: Names of the features in the dataset, used for visualization labeling.
    - all_predictions: Prediction outcomes from the model for the test dataset.
    - all_predictions_class, all_labels_class: Classified prediction results and actual labels for the test dataset.
    - device: The computation device ('cuda' or 'cpu') for PyTorch operations.
    - save_path: Directory path where the generated visualization plots will be stored.

    Workflow:
    1. Prepares the outlier samples from the test dataset for the IMV-LSTM model input.
    2. Computes SHAP values for these outlier samples to analyze feature contributions.
    3. Evaluates the model to extract alpha (time-step importance) and beta (feature importance) values for the outlier samples.
    4. Generates and saves visualizations for each outlier sample, including:
       - Heatmaps showing the importance of features across different timesteps.
       - Bar plots indicating the overall importance of each feature.
       - Decision plots and force plots derived from SHAP values to illustrate feature contributions to the prediction.
       - Image plots for a visual summary of feature impacts.
    5. Ensures the output directory is created if it does not exist.
    6. Outputs the prediction details (predicted value, predicted and true labels) for each outlier sample.

    This function provides a comprehensive analysis of outlier predictions in IMV-LSTM models, helping to understand the
    temporal and feature-specific influences on model performance and prediction anomalies.
    """
    outlier_elements_array = np.array(outlier_elements)
    outlier_sample = [X_test_scaled[i] for i in outlier_elements_array]

    # Calculate SHAP values for the outlier samples
    shap_values = explainer.shap_values(torch.tensor(outlier_sample, dtype=torch.float32).to(device),  check_additivity=False)

    # Set the model to evaluation mode and calculate alphas and betas
    model.eval()
    sample_alphas = []
    sample_betas = []

    with torch.no_grad():
        for sample in outlier_sample:
            _, alpha, beta = model(torch.tensor([sample]).float().to(device))
            sample_alphas.append(alpha.cpu().numpy())
            sample_betas.append(beta.cpu().numpy())

    alphas_sample = np.concatenate(sample_alphas)
    betas_sample = np.concatenate(sample_betas)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Visualize the importance of features and timesteps using alpha values
    for sample_num in range(len(outlier_elements)):
        fig, ax = plt.subplots(figsize=(24, 20))
        im = ax.imshow(alphas_sample[sample_num].squeeze().transpose(1,0))

        # 틱 라벨 설정
        ax.set_xticks(np.arange(np.array(X_train_scaled).shape[1]))
        ax.set_yticks(np.arange(len(X_train[0].columns)))
        ax.set_xticklabels(["t-"+str(i+1) for i in np.arange(np.array(X_train_scaled).shape[1]-1, -1, -1)], fontsize=12)
        ax.set_yticklabels(list(X_train[0].columns), fontsize=12)

        # 셀 값 표시
        for i in range(len(X_train[0].columns)):
            for j in range(np.array(X_train_scaled).shape[1]):
                text = ax.text(j, i, round(alphas_sample[sample_num].squeeze().transpose(1,0)[i, j], 3),
                            ha="center", va="center", color="w")
        
        # 색상 막대 추가
        fig.colorbar(im, ax=ax)

        ax.set_title(f"Temporal level attention map for sample {outlier_elements[sample_num]}", fontsize=20)
        fig.tight_layout()
        plt.savefig(f'{save_path}/IMV_Temporal_level_attention_map_{outlier_elements[sample_num]}.png', dpi=300,  bbox_inches='tight')
        plt.clf()
        
    # the feature importance         
        feature_names = X_train[0].columns.tolist()

        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': betas_sample[sample_num].squeeze()
        })

        feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        top_feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(10)
        plt.figure(figsize=(20, 20))
        top_feature_importances.plot(kind='bar', x='feature', y='importance', legend=False)
        plt.title('Top 10 Feature Importances using variable level attention')
        plt.ylabel('Average Impact on Model Output')
        fig.tight_layout()  
        plt.savefig(f'{save_path}/IMV_feature_importance_{outlier_elements[sample_num]}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    # decision plots
        shap.decision_plot(explainer.expected_value, np.sum(shap_values[sample_num], axis=0), features=feature_names, show=False)
        plt.savefig(f"{save_path}/decision_plot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()

    # force plots
        shap.force_plot(explainer.expected_value, np.sum(shap_values[sample_num], axis=0), features=feature_names, matplotlib=True, show=False)
        plt.savefig(f"{save_path}/force_plot_{outlier_elements[sample_num]}.png", dpi=300, bbox_inches='tight')
        plt.clf()
    
    # image plots
        shap_image_plot(np.expand_dims(shap_values[sample_num].transpose(), axis=-1), -np.expand_dims(np.array(outlier_sample[0]).transpose(), -1), show=False, width=100, aspect=0.05, hspace=0.05 , features=feature_names)
        plt.savefig(f"{save_path}/imageplot_{outlier_elements[sample_num]}.png",dpi=300, bbox_inches='tight')
        plt.clf()

    # Print predictions and true labels for outliers
    for idx in outlier_elements:
        predicted_value = all_predictions[idx]  # The actual predicted value from the model
        predicted_label = all_predictions_class[idx]  # The predicted class label
        true_label = all_labels_class[idx]  # The true class label
        print(f"Index: {idx}, Predicted Value: {predicted_value}, Predicted Label: {predicted_label}, True Label: {true_label}")

def imvlstm_classification_outlier_visualization(outlier_elements, model,explainer, X_test_scaled,X_train_scaled,X_train,feature_names, all_predictions, all_labels, device, save_path='IMV_LSTM_clf/outlier'):
    """
    Visualizes outliers in IMV-LSTM regression model predictions using SHAP values and model-specific outputs (alphas and betas).
    It creates various plots to explore the impact of individual features and timesteps on the anomalous predictions.

    Parameters:
    - outlier_elements: Indices of the outlier samples in the test dataset.
    - model: The IMV-LSTM model used for regression predictions.
    - explainer: A SHAP DeepExplainer object initialized with the model and a background dataset for SHAP value computation.
    - X_test_scaled: Scaled test dataset used for model predictions.
    - X_train_scaled, X_train: Scaled and original training datasets, used for feature and timestep visualization.
    - feature_names: Names of the features in the dataset, used for visualization labeling.
    - all_predictions: Prediction outcomes from the model for the test dataset.
    - all_predictions_class, all_labels_class: Classified prediction results and actual labels for the test dataset.
    - device: The computation device ('cuda' or 'cpu') for PyTorch operations.
    - save_path: Directory path where the generated visualization plots will be stored.

    Workflow:
    1. Prepares the outlier samples from the test dataset for the IMV-LSTM model input.
    2. Computes SHAP values for these outlier samples to analyze feature contributions.
    3. Evaluates the model to extract alpha (time-step importance) and beta (feature importance) values for the outlier samples.
    4. Generates and saves visualizations for each outlier sample, including:
       - Heatmaps showing the importance of features across different timesteps.
       - Bar plots indicating the overall importance of each feature.
       - Decision plots and force plots derived from SHAP values to illustrate feature contributions to the prediction.
       - Image plots for a visual summary of feature impacts.
    5. Ensures the output directory is created if it does not exist.
    6. Outputs the prediction details (predicted value, predicted and true labels) for each outlier sample.

    This function provides a comprehensive analysis of outlier predictions in IMV-LSTM models, helping to understand the
    temporal and feature-specific influences on model performance and prediction anomalies.
    """
    outlier_elements_array = np.array(outlier_elements)
    outlier_sample = [X_test_scaled[i] for i in outlier_elements_array]
    outlier_prediction = [int(all_predictions[i]) for i in outlier_elements_array]

    shap_values = explainer.shap_values(torch.tensor(outlier_sample, dtype=torch.float32).to(device), check_additivity=False)
    
    # Set model to evaluation mode
    model.eval()
    sample_alphas = []
    sample_betas = []

    with torch.no_grad():
        for sample in outlier_sample:
            outputs, alpha, beta = model(torch.tensor([sample]).float().to(device))
            sample_alphas.append(alpha.cpu().numpy())
            sample_betas.append(beta.cpu().numpy())
    
    alphas_sample = np.concatenate(sample_alphas)
    betas_sample = np.concatenate(sample_betas)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for sample_num in range(len(outlier_elements)):
        fig, ax = plt.subplots(figsize=(24, 20))
        im = ax.imshow(alphas_sample[sample_num].squeeze().transpose(1,0))

        # 틱 라벨 설정
        ax.set_xticks(np.arange(np.array(X_train_scaled).shape[1]))
        ax.set_yticks(np.arange(len(X_train[0].columns)))
        ax.set_xticklabels(["t-"+str(i+1) for i in np.arange(np.array(X_train_scaled).shape[1]-1, -1, -1)], fontsize=12)
        ax.set_yticklabels(list(X_train[0].columns), fontsize=12)

        # 셀 값 표시
        for i in range(len(X_train[0].columns)):
            for j in range(np.array(X_train_scaled).shape[1]):
                text = ax.text(j, i, round(alphas_sample[sample_num].squeeze().transpose(1,0)[i, j], 3),
                            ha="center", va="center", color="w")
        
        # 색상 막대 추가
        fig.colorbar(im, ax=ax)

        ax.set_title(f"Temporal level attention map for sample {outlier_elements[sample_num]}", fontsize=20)
        fig.tight_layout()        
        plt.savefig(f'{save_path}/IMV_Temporal_level_attention_map_{outlier_elements[sample_num]}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    # the feature importance 
        feature_names = X_train[0].columns.tolist()

        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': betas_sample[sample_num].squeeze()
        })

        feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        top_feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(10)

        plt.figure(figsize=(20, 20))
        top_feature_importances.plot(kind='bar', x='feature', y='importance', legend=False)
        plt.title('Top 10 Feature Importances using variable level attention')
        plt.ylabel('Average Impact on Model Output')
        fig.tight_layout()
        plt.savefig(f'{save_path}/IMV_feature_importance_{outlier_elements[sample_num]}.png',dpi=300, bbox_inches='tight')
        plt.clf()

    # decision plots
        shap.decision_plot(explainer.expected_value[outlier_prediction[sample_num]],np.sum(shap_values[outlier_prediction[sample_num]][sample_num], axis=0), features=feature_names, show=False)
        plt.savefig(f"{save_path}/decision_plot_{outlier_elements[sample_num]}.png",dpi=300, bbox_inches='tight')
        plt.clf()

    # force plots
        shap.force_plot(explainer.expected_value[outlier_prediction[sample_num]], np.sum(shap_values[outlier_prediction[sample_num]][sample_num], axis=0), features=feature_names , matplotlib=True,show=False)
        plt.savefig(f"{save_path}/force_plot_{outlier_elements[sample_num]}.png",dpi=300, bbox_inches='tight')
        plt.clf()

    # image plots
        shap_image_plot(np.expand_dims(shap_values[outlier_prediction[sample_num]][sample_num].transpose(), axis=-1),
                     -np.expand_dims(np.array(X_test_scaled[0]).transpose(), -1), show=False, width=100, aspect=0.05, hspace=0.05 , features=feature_names)
        plt.savefig(f"{save_path}/imageplot_{outlier_elements[sample_num]}.png",dpi=300, bbox_inches='tight')
        plt.clf()

    for idx in outlier_elements:
        predicted_value = all_predictions[idx]
        true_label = all_labels[idx]
        print(f"Index: {idx}, Predicted Value: {predicted_value}, True Label: {true_label}")


