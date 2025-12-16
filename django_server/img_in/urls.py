from django.urls import path

from . import views

urlpatterns = [
    path("translate/", views.TranslatorView.as_view(), name="translate"),
    path("feedback/", views.FeedbackView.as_view(), name="feedback"),
]
