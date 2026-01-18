from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="TrainingFeedback",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("image", models.ImageField(upload_to="feedback_images/")),
                ("predicted_label", models.CharField(max_length=1)),
                ("correct_label", models.CharField(max_length=1)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-timestamp"],
            },
        ),
    ]
