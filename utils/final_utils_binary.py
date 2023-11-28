import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_accuracy_loss(model_name, model_parameter, accuracies, checkpoint_losses):
    if not model_parameter:
        out_string = f"{model_name}_AccuLoss.png"
    else:
        out_string = f"{model_name}_{model_parameter}_AccuLoss.png"

    color_cycle = plt.rcParams['axes.prop_cycle']()

    fig_losses, ax_losses = plt.subplots(dpi=100)
    ax_losses.plot(checkpoint_losses[0], label="Train Loss", **next(color_cycle))
    ax_losses.plot(checkpoint_losses[1], label="Validation Loss", **next(color_cycle))
    ax_losses.set_xlabel('Epochs')
    ax_losses.set_ylabel('Loss')

    ax_accuracy = ax_losses.twinx()
    ax_accuracy.plot(accuracies[1], label="Validation Accuracy", **next(color_cycle))
    ax_accuracy.plot(accuracies[0], label="Train Accuracy", **next(color_cycle))
    ax_accuracy.set_ylabel('Accuracy')

    handles, labels = [(a + b) for a, b in zip(ax_losses.get_legend_handles_labels(), ax_accuracy.get_legend_handles_labels())]
    fig_losses.legend(handles, labels, loc='center')
    plt.savefig(out_string)
    plt.close(fig_losses)


# Get the convolutional layers in the model
def get_conv_layers(model_children) -> list:
    # Save the conv layers to this list
    conv_layers = []
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children) - 1):
        if len(list(model_children[i].children()))> 0:
            conv_layers.append(list(model_children[i].children())[0])
    return conv_layers


def plot_feature_maps(input_dataloader, conv_layers: list, device):
    data_iter = iter(input_dataloader)

    images = next(data_iter)

    image = images['image'][0].to(device)

    outputs = []
    processed = []
    names = []

    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        init_gray_scale = torch.sum(feature_map,0)
        gray_scale = init_gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i], cmap='rainbow')
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)

    return fig


def plot_batch_features(input_dataloader, conv_layers: list, device, batch_size, label_names, variance = 0):
    data_iter = iter(input_dataloader)

    # Add some variance
    if variance == 0:
        images = next(data_iter)
    elif variance == 1:
        _ = next(data_iter)
        images = next(data_iter)
    else:
        _ = next(data_iter)
        _ = next(data_iter)
        images = next(data_iter)

    outputs = []
    names = []
    processed = []

    for tracker in range(batch_size):
        labels = images['labels'][tracker]
        plot_image = images['image'][tracker].to(device)

        label = label_names[labels]
        j = 0

        for layer in conv_layers:
            j += 1
            plot_image = layer(plot_image)
            # Append the results to the output list
            outputs.append(plot_image)
            # Create a name for the plot for the current convolution
            image_name = f"conv{j}_{str(label)}"
            # Append the name to the output name list
            names.append(image_name)

    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        init_gray_scale = torch.sum(feature_map, 0)
        gray_scale = init_gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 50))

    for i in range(len(processed)):
        if i < 20:
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(processed[i], cmap='rainbow')
            a.axis("off")
            a.set_title(names[i], fontsize=30)


# Activation criterion function
def criterion(loss_func, prediction, batch_labels):
    losses = 0
    losses += loss_func(prediction, batch_labels)
    return losses


# Training function
def train(train_dataloader, validation_dataloader, input_model, optimizer, loss_func, scheduler, number_epochs, device, use_scheduler=False, lambda_factor=0.0):
    # Define starting variables
    train_losses = []
    validation_losses = []
    accuracies = [[], []]
    checkpoint_losses = [[], []]
    train_precision_metrics = []
    test_precision_metrics = []
    train_recall_metrics = []
    test_recall_metrics = []
    train_f1_scores = []
    test_f1_scores = []

    # Get the number of steps for each phase
    train_total_steps = len(train_dataloader)
    validation_total_steps = len(validation_dataloader)

    # Create regularization function
    if lambda_factor > 0.0:
        l_penalty = torch.nn.L1Loss(reduction='sum')

    for epoch in range(number_epochs):
        # Define starting variables
        n_samples = 0
        n_correct = 0
        train_target_true = 0
        train_predicted_true = 0
        train_correct_true = 0
        test_target_true = 0
        test_predicted_true = 0
        test_correct_true = 0
        # Set the model to training mode
        input_model.train()
        for t_index, t_x in enumerate(train_dataloader):
            # Reset the gradients
            optimizer.zero_grad()
            # Get the data and labels
            batch_images = t_x['image'].to(device)
            batch_labels = (t_x['labels'].float()).to(device)
            # Get the prediction
            pred = input_model(batch_images)
            pred = torch.squeeze(pred, 1)
            # Calculate accuracy
            predicted = (pred > 0.5).float()
            n_correct += (predicted == batch_labels).sum().item()
            # Save the number of items in this batch
            n_samples += batch_labels.size(0)
            # Calculate loss and update parameters
            loss = criterion(loss_func, pred, batch_labels)
            # L1 and L2 regularization
            if lambda_factor > 0.0:
                reg_loss = l_penalty(pred, batch_labels)
                loss += lambda_factor * reg_loss
            # Save the losses
            train_losses.append(loss.item())
            # Update the weights
            loss.backward()
            optimizer.step()
            # Calculate metrics
            train_target_true += float((batch_labels == 1.).sum().item())
            train_predicted_true += float((predicted).sum().item())
            train_correct_true += torch.sum((predicted == batch_labels) * (predicted == 1.).float())
            # If it is the last data in the dataloader...
            if (t_index+1) % (int(train_total_steps/1)) == 0:
                # Starting variables
                train_checkpoint_accuracy = 0
                # Calculate and store loss
                train_checkpoint_loss = torch.tensor(train_losses).mean().item()
                checkpoint_losses[0].append(train_checkpoint_loss)
                # Calculate accuracy
                train_checkpoint_accuracy = 100.0 * n_correct / n_samples
                # Store accuracy
                accuracies[0].append(train_checkpoint_accuracy)
        if use_scheduler:
            # Use the scheduler when designating scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
        # Set some variables for evaluation
        n_samples = 0
        n_correct = 0
        # Set the model to validation mode
        input_model.eval()
        for index, x in enumerate(validation_dataloader):
            # Get the data and labels
            batch_images = x['image'].to(device)
            batch_labels = (x['labels'].float()).to(device)
            # Get the prediction
            pred = input_model(batch_images)
            pred = torch.squeeze(pred, 1)
            # Calculate accuracy
            predicted = (pred > 0.5).float()
            n_correct += (predicted == batch_labels).sum().item()
            # Save the number of items in this batch
            n_samples += batch_labels.size(0)
            # Calculate loss
            loss = criterion(loss_func, pred, batch_labels)
            # Save the losses
            validation_losses.append(loss.item())
            # Calculate metrics
            test_target_true += float((batch_labels == 1.).sum().item())
            test_predicted_true += float((predicted).sum().item())
            test_correct_true += torch.sum((predicted == batch_labels) * (predicted == 1.).float())
            # If it is the last data in the dataloader...
            if (index+1) % (int(validation_total_steps/1)) == 0:
                # Starting variables
                validation_checkpoint_accuracy = 0
                # Calculate and store loss
                validation_checkpoint_loss = torch.tensor(validation_losses).mean().item()
                checkpoint_losses[1].append(validation_checkpoint_loss)
                # Calculate accuracy
                validation_checkpoint_accuracy = 100.0 * n_correct / n_samples
                # Store accuracy
                accuracies[1].append(validation_checkpoint_accuracy)
        # If it is the first epoch, skip the model saving check
        if epoch == 0:
            # Get the current loss sum and set it as the absolute low
            abs_low = train_checkpoint_loss + validation_checkpoint_loss
        else:
            # Get the current loss sum
            cur_low = train_checkpoint_loss + validation_checkpoint_loss
            # If the lowest recorded loss is higher than the current loss
            if abs_low > cur_low:
                torch.save(input_model.state_dict(), 'single_class_model_weights.pth')
                torch.save(input_model, 'single_class_model.pth')
                abs_low = cur_low
        # Calculate epoch metrics
        train_recall = train_correct_true / train_target_true
        train_precision = train_correct_true / train_predicted_true
        train_f1_score = 2 * train_precision * train_recall / (train_precision + train_recall)
        test_recall = test_correct_true / test_target_true
        test_precision = test_correct_true / test_predicted_true
        test_f1_score = 2 * test_precision * test_recall / (test_precision + test_recall)
        train_precision_metrics.append(train_precision.detach().cpu().numpy())
        test_precision_metrics.append(test_precision.detach().cpu().numpy())
        train_recall_metrics.append(train_recall.detach().cpu().numpy())
        test_recall_metrics.append(test_recall.detach().cpu().numpy())
        train_f1_scores.append(train_f1_score.detach().cpu().numpy())
        test_f1_scores.append(test_f1_score.detach().cpu().numpy())
        # Print the epoch statistics
        print(f"Epoch [{epoch + 1}/{number_epochs}]:\n"
              f"Train accuracy: {train_checkpoint_accuracy:.4f}%, Validation accuracy: {validation_checkpoint_accuracy:.4f}%. \n"
              f"Training loss: {train_checkpoint_loss:.4f}, Validation loss: {validation_checkpoint_loss:.4f}. \n"
              f"Training precision: {train_precision:.4f}, Validation precision: {test_precision:.4f}. \n"
              f"Training recall: {train_recall:.4f}, Validation recall: {test_recall:.4f}. \n"
              f"Training F1 score: {train_f1_score:.4f}, Validation F1 score: {test_f1_score:.4f}. \n")

    return accuracies, checkpoint_losses, train_precision_metrics, test_precision_metrics, train_recall_metrics, test_recall_metrics, train_f1_scores, test_f1_scores


# Validation function
def make_predictions(input_dataloader, input_model, device):
    all_predictions = torch.tensor([]).to(device)
    all_true_labels = torch.tensor([]).to(device)

    with torch.no_grad():
        y_pred = []
        y_true = []

        for graph_images in input_dataloader:
            input_images = graph_images['image'].to(device)
            predictions = input_model(input_images)

            predicted = (predictions > 0.5).float().data.cpu().numpy()
            y_pred.extend(predicted)
            labels = graph_images['labels'].data.cpu().numpy()
            y_true.extend(labels)

        cfx_matrix = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cfx_matrix).plot(cmap='viridis')


def plot_weights(conv_layers):
    for conv_layer in conv_layers:
        # getting the weight tensor data
        weight_tensor = conv_layer.weight.data
        # kernels depth * number of kernels
        n_plots = weight_tensor.shape[0] * weight_tensor.shape[1]
        n_cols = 12
        n_rows = 1 + n_plots // n_cols

        # convert tensor to numpy image
        n_pimg = np.array(weight_tensor.cpu().numpy(), np.float32)

        count = 0
        fig = plt.figure(figsize=(n_cols, n_rows))

        # looping through all the kernels in each channel
        for i in range(weight_tensor.shape[0]):
            for j in range(weight_tensor.shape[1]):
                count += 1
                ax1 = fig.add_subplot(n_rows, n_cols, count)
                n_pimg = np.array(weight_tensor[i, j].cpu().numpy(), np.float32)
                n_pimg = (n_pimg - np.mean(n_pimg)) / np.std(n_pimg)
                n_pimg = np.minimum(1, np.maximum(0, (n_pimg + 0.5)))
                ax1.imshow(n_pimg)
                ax1.set_title(str(i) + ',' + str(j))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

        plt.tight_layout()
        plt.show()

