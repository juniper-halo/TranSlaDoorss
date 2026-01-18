from django.db import models


class TrainingFeedback(models.Model):
    """Store user feedback on predictions to improve model training."""

    image = models.ImageField(upload_to="feedback_images/")
    predicted_label = models.CharField(max_length=1)
    correct_label = models.CharField(max_length=1)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"Feedback: {self.predicted_label} -> {self.correct_label} ({self.timestamp})"
