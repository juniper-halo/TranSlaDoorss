from django.apps import AppConfig


class ImgInConfig(AppConfig):
    # keep default config so Django auto-discovers the app
    default_auto_field = "django.db.models.BigAutoField"
    name = "img_in"
