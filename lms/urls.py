from django.urls import path
from django.views.generic import TemplateView

from . import views


urlpatterns = [
    path("", views.course_list, name="course_list"),
    path("recommendations/", views.recommend_courses, name="recommend_courses"),
    path("courses/upload/", views.course_upload, name="course_upload"),
    path(
        "courses/<int:course_id>/lessons/upload/",
        views.video_lesson_upload,
        name="video_lesson_upload",
    ),
    path(
        "lessons/<int:lesson_id>/",
        views.video_lesson_detail,
        name="video_lesson_detail",
    ),
    path(
        "lessons/<int:lesson_id>/quiz/generate/",
        views.generate_quiz,
        name="generate_quiz",
    ),
    path("quizzes/<int:quiz_id>/submit/", views.quiz_submit, name="quiz_submit"),
    path(
        "quiz-attempts/<int:attempt_id>/results/",
        views.quiz_results,
        name="quiz_results",
    ),
    path("courses/<int:course_id>/", views.course_detail, name="course_detail"),
    path(
        "courses/<int:course_id>/users/<int:user_id>/certificate/",
        views.generate_certificate,
        name="generate_certificate",
    ),
    path(
        "courses/upload/success/",
        TemplateView.as_view(template_name="lms/course_upload_success.html"),
        name="course_upload_success",
    ),
]
