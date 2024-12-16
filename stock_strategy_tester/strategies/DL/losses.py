import torch
import torch.nn as nn


class TrendAwareLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Custom loss function combining L1 loss and Trend Classification Metric.

        Args:
            alpha (float): Weight for the trend classification component (0 <= alpha <= 1).
                           L1 loss will have a weight of (1 - alpha).
        """
        super(TrendAwareLoss, self).__init__()
        self.alpha = alpha
        # self.l1_loss = nn.L1Loss()
        self.l1_loss = nn.MSELoss()

    def trend_classification_metric(self, y_pred, y_true):
        """
        Calculate trend classification accuracy.
        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): Ground truth values.
        Returns:
            Tensor: Trend classification accuracy (0-1 scale).
        """
        pred_trend = torch.sign(y_pred)
        true_trend = torch.sign(y_true)
        correct_trends = (pred_trend == true_trend).float()
        return correct_trends.sum()

    def diversity_loss(self, y_pred, y_true):
        """
        Calculate diversity loss.
        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): Ground truth values.
        """
        # Calculate the mean of the predicted values
        pred_mean = torch.mean(y_pred, dim=0)
        # Calculate mean for each true value
        true_mean = torch.mean(y_true, dim=0)
        # Calculate the diversity loss
        true_diversity_loss = (torch.mean(abs(true_mean - y_true)))
        pred_diversity_loss = (torch.mean(abs(pred_mean - y_pred)))
        diversity_loss = abs(true_diversity_loss - pred_diversity_loss) ** -0.5
        return diversity_loss

    def forward(self, y_pred, y_true):
        """
        Calculate the combined loss.
        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): Ground truth values.
        Returns:
            Tensor: Combined loss value.
        """
        # Calculate L1 Loss
        l1_loss_value = self.l1_loss(y_pred, y_true)

        # Calculate diversity loss
        diversity_loss = self.diversity_loss(y_pred, y_true)

        # Calculate Trend Metric (invert accuracy to represent a loss)
        trend_accuracy = self.trend_classification_metric(y_pred, y_true)
        trend_loss = 1 - trend_accuracy

        # Combine losses
        combined_loss = 10*2 * ( (1 - self.alpha) * l1_loss_value + self.alpha * trend_loss ) + diversity_loss
        return combined_loss