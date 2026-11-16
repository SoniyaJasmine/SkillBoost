import asyncio
import json
import os
from datetime import datetime, timezone as datetime_timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from asgiref.sync import async_to_sync
from django.http import Http404
from django.test import RequestFactory, SimpleTestCase, TestCase
from django.urls import reverse

from .forms import CourseForm, VideoLessonForm
from .models import (
    Answer,
    Course,
    CourseCompletion,
    Question,
    Quiz,
    QuizAttempt,
    VideoLesson,
)
from .views import (
    _request_course_recommendations,
    _request_gemini,
    _save_quiz_questions,
    course_detail,
    generate_certificate,
    quiz_results,
    quiz_submit,
    recommend_courses,
)


class CourseFormTests(SimpleTestCase):
    def test_form_accepts_course_fields(self):
        form = CourseForm(
            data={"title": "Django Basics", "description": "An introduction to Django."}
        )

        self.assertTrue(form.is_valid())

    def test_form_rejects_missing_required_fields(self):
        form = CourseForm(data={"title": "", "description": ""})

        self.assertFalse(form.is_valid())
        self.assertIn("title", form.errors)
        self.assertIn("description", form.errors)


class CourseUploadViewTests(SimpleTestCase):
    def test_get_renders_blank_form(self):
        response = self.client.get(reverse("course_upload"))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["form"].is_bound)
        self.assertContains(response, 'name="csrfmiddlewaretoken"')

    def test_valid_post_saves_and_redirects_to_success_page(self):
        with patch("lms.views.CourseForm") as form_class:
            form = form_class.return_value
            form.is_valid.return_value = True

            response = self.client.post(
                reverse("course_upload"),
                {"title": "Django Basics", "description": "An introduction to Django."},
            )

        form.save.assert_called_once_with()
        self.assertRedirects(response, reverse("course_upload_success"))

    def test_invalid_post_renders_form_without_saving(self):
        with patch("lms.views.CourseForm") as form_class:
            form = form_class.return_value
            form.is_valid.return_value = False

            response = self.client.post(
                reverse("course_upload"),
                {"title": "", "description": "An introduction to Django."},
            )

        self.assertEqual(response.status_code, 200)
        form.save.assert_not_called()


class CourseListViewTests(SimpleTestCase):
    def test_homepage_lists_courses_and_shows_navigation(self):
        course = SimpleNamespace(
            pk=7,
            title="Django Basics",
            description="An introduction to Django.",
        )
        with patch("lms.views.Course.objects.order_by", return_value=[course]):
            response = self.client.get(reverse("course_list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SkillBoost")
        self.assertContains(response, "Recommended for You")
        self.assertContains(response, "Upload a course")
        self.assertContains(response, "/static/lms/style.css?v=20261109-nav")
        self.assertContains(response, course.title)
        self.assertContains(
            response,
            reverse("course_detail", kwargs={"course_id": course.pk}),
        )

    def test_homepage_renders_personalized_recommendations_for_matching_courses(self):
        course = SimpleNamespace(
            pk=7,
            title="Django Basics",
            description="An introduction to Django.",
        )
        request = RequestFactory().get("/")
        request.user = SimpleNamespace(is_authenticated=True, pk=12)

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("lms.views.Course.objects.order_by", return_value=[course]),
            patch(
                "lms.views._get_user_learning_activity",
                return_value=([{"title": "Python"}], [{"score": 4}]),
            ) as get_activity,
            patch(
                "lms.views._request_course_recommendations",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "title": " django basics ",
                        "description": "Build on your Python experience.",
                        "relevance_score": 91,
                    },
                    {
                        "title": "Unavailable course",
                        "description": "Not in the LMS catalog.",
                        "relevance_score": 80,
                    },
                ],
            ),
        ):
            response = recommend_courses(request)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Recommended for You")
        self.assertContains(response, "Build on your Python experience.")
        self.assertContains(response, "Relevance score: 91")
        self.assertContains(
            response,
            reverse("course_detail", kwargs={"course_id": course.pk}),
        )
        self.assertNotContains(response, "Unavailable course")
        get_activity.assert_called_once_with(request.user.pk)

    def test_recommendation_request_uses_grounding_and_activity_prompt(self):
        response_body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        [
                                            {
                                                "title": "Django Basics",
                                                "description": "Learn Django.",
                                                "relevance_score": 90,
                                            }
                                        ]
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode()
        available_courses = [{"title": "Django Basics", "description": "Django"}]
        completed_courses = [{"title": "Python"}]
        quiz_results = [{"course": "Python", "score": 4}]

        with patch(
            "lms.views._send_gemini_request", return_value=response_body
        ) as send_request:
            recommendations = async_to_sync(_request_course_recommendations)(
                "test-key", available_courses, completed_courses, quiz_results
            )

        self.assertEqual(recommendations[0]["relevance_score"], 90)
        payload = send_request.call_args.args[1]
        self.assertEqual(payload["tools"], [{"google_search": {}}])
        self.assertEqual(
            payload["generationConfig"]["responseSchema"]["items"]["required"],
            ["title", "description", "relevance_score"],
        )
        prompt = payload["contents"][0]["parts"][0]["text"]
        self.assertIn("Django Basics", prompt)
        self.assertIn("Python", prompt)
        self.assertIn("quiz performance", prompt)


class VideoLessonFormTests(SimpleTestCase):
    def test_form_accepts_video_lesson_fields(self):
        form = VideoLessonForm(
            data={
                "title": "Introduction",
                "video_url": "https://example.com/lesson",
                "description": "The first lesson.",
            }
        )

        self.assertTrue(form.is_valid())

    def test_form_rejects_invalid_video_url_and_missing_fields(self):
        form = VideoLessonForm(
            data={
                "title": "",
                "video_url": "not a URL",
                "description": "",
            }
        )

        self.assertFalse(form.is_valid())
        self.assertIn("title", form.errors)
        self.assertIn("video_url", form.errors)
        self.assertIn("description", form.errors)


class ModelCRUDTests(TestCase):
    def setUp(self):
        from django.contrib.auth import get_user_model

        self.user = get_user_model().objects.create_user(
            username="model-test-user",
            password=None,
        )
        self.course = Course.objects.create(
            title="Python Basics",
            description="Learn Python.",
        )
        self.lesson = VideoLesson.objects.create(
            course=self.course,
            title="Introduction",
            video_url="https://example.com/lesson",
            description="Getting started.",
        )
        self.quiz = Quiz.objects.create(course=self.course, lesson=self.lesson)
        self.question = Question.objects.create(
            quiz=self.quiz,
            question_text="What is Python?",
        )

    def test_course_can_be_created_updated_and_deleted(self):
        self.assertEqual(str(self.course), "Python Basics")

        self.course.title = "Python Fundamentals"
        self.course.save()
        self.course.refresh_from_db()
        self.assertEqual(self.course.title, "Python Fundamentals")

        self.course.delete()
        self.assertFalse(Course.objects.filter(pk=self.course.pk).exists())

    def test_video_lesson_can_be_created_updated_and_deleted(self):
        self.assertEqual(str(self.lesson), "Introduction")

        self.lesson.title = "Python Introduction"
        self.lesson.save()
        self.lesson.refresh_from_db()
        self.assertEqual(self.lesson.title, "Python Introduction")

        self.lesson.delete()
        self.assertFalse(VideoLesson.objects.filter(pk=self.lesson.pk).exists())

    def test_quiz_can_be_created_updated_and_deleted(self):
        self.assertEqual(str(self.quiz), f"Quiz for {self.lesson}")

        self.quiz.is_ai_generated = True
        self.quiz.save()
        self.quiz.refresh_from_db()
        self.assertTrue(self.quiz.is_ai_generated)

        self.quiz.delete()
        self.assertFalse(Quiz.objects.filter(pk=self.quiz.pk).exists())

    def test_question_can_be_created_updated_and_deleted(self):
        self.assertEqual(str(self.question), "What is Python?")

        self.question.question_text = "What does Python run on?"
        self.question.save()
        self.question.refresh_from_db()
        self.assertEqual(self.question.question_text, "What does Python run on?")

        self.question.delete()
        self.assertFalse(Question.objects.filter(pk=self.question.pk).exists())

    def test_answer_can_be_created_updated_and_deleted(self):
        answer = Answer.objects.create(
            question=self.question,
            answer_text="A programming language",
            is_correct=True,
        )
        self.assertEqual(str(answer), "A programming language")

        answer.answer_text = "A programming language and runtime"
        answer.save()
        answer.refresh_from_db()
        self.assertEqual(answer.answer_text, "A programming language and runtime")

        answer.delete()
        self.assertFalse(Answer.objects.filter(pk=answer.pk).exists())

    def test_quiz_attempt_can_be_created_updated_and_deleted(self):
        attempt = QuizAttempt.objects.create(
            user=self.user,
            quiz=self.quiz,
            score=1,
            results={"1": {"is_correct": True}},
        )
        self.assertIn("1", str(attempt))

        attempt.score = 2
        attempt.results = {"1": {"is_correct": True}, "2": {"is_correct": True}}
        attempt.save()
        attempt.refresh_from_db()
        self.assertEqual(attempt.score, 2)
        self.assertEqual(len(attempt.results), 2)

        attempt.delete()
        self.assertFalse(QuizAttempt.objects.filter(pk=attempt.pk).exists())

    def test_course_completion_can_be_created_updated_and_deleted(self):
        completion = CourseCompletion.objects.create(
            user=self.user,
            course=self.course,
        )
        self.assertIn("False", str(completion))

        completion.is_completed = True
        completion.save()
        completion.refresh_from_db()
        self.assertTrue(completion.is_completed)

        completion.delete()
        self.assertFalse(CourseCompletion.objects.filter(pk=completion.pk).exists())


class VideoLessonUploadViewTests(SimpleTestCase):
    @patch("lms.views.get_object_or_404")
    def test_get_renders_course_context_and_blank_form(self, get_course):
        course = SimpleNamespace(pk=7, title="Django Basics")
        get_course.return_value = course

        response = self.client.get(
            reverse("video_lesson_upload", kwargs={"course_id": course.pk})
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["course"], course)
        self.assertFalse(response.context["form"].is_bound)
        self.assertContains(response, course.title)
        self.assertContains(response, 'name="csrfmiddlewaretoken"')

    @patch("lms.views.get_object_or_404")
    @patch("lms.views.VideoLessonForm")
    def test_valid_post_saves_lesson_to_course_and_redirects(
        self, form_class, get_course
    ):
        course = SimpleNamespace(pk=7)
        course.videolesson_set = SimpleNamespace(all=lambda: [])
        get_course.return_value = course
        form = form_class.return_value
        form.is_valid.return_value = True
        lesson = form.save.return_value

        response = self.client.post(
            reverse("video_lesson_upload", kwargs={"course_id": course.pk}),
            {
                "title": "Introduction",
                "video_url": "https://example.com/lesson",
                "description": "The first lesson.",
            },
        )

        self.assertIs(lesson.course, course)
        lesson.save.assert_called_once_with()
        self.assertRedirects(
            response, reverse("course_detail", kwargs={"course_id": course.pk})
        )

    @patch("lms.views.get_object_or_404")
    def test_course_detail_lists_lessons_and_links_to_upload(self, get_course):
        course = SimpleNamespace(
            pk=7,
            title="Django Basics",
            description="An introduction to Django.",
        )
        course.videolesson_set = SimpleNamespace(all=lambda: [])
        get_course.return_value = course

        response = self.client.get(
            reverse("course_detail", kwargs={"course_id": course.pk})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, course.title)
        self.assertContains(
            response,
            reverse("video_lesson_upload", kwargs={"course_id": course.pk}),
        )
        self.assertNotContains(response, "Download Certificate")


class QuizGenerationTests(SimpleTestCase):
    @patch("lms.views.get_object_or_404")
    def test_video_lesson_detail_shows_generate_button_and_success(self, get_lesson):
        lesson = SimpleNamespace(
            pk=9,
            title="Django lesson",
            description="A lesson about Django.",
            course=SimpleNamespace(title="Django course"),
            quizzes=SimpleNamespace(
                filter=lambda **_kwargs: SimpleNamespace(
                    prefetch_related=lambda *_args: []
                )
            ),
            video_url="https://example.com/lesson",
        )
        get_lesson.return_value = lesson

        response = self.client.get(
            reverse("video_lesson_detail", kwargs={"lesson_id": lesson.pk})
            + "?quiz_generated=1"
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Quiz generated successfully.")
        self.assertContains(
            response,
            f'action="{reverse("generate_quiz", kwargs={"lesson_id": lesson.pk})}"',
        )
        self.assertContains(response, "Generate quiz")

    @patch("lms.views.get_object_or_404")
    def test_video_lesson_page_displays_saved_question_and_answers(self, get_lesson):
        answers = [
            SimpleNamespace(pk=11, answer_text="Option one"),
            SimpleNamespace(pk=12, answer_text="Option two"),
        ]
        question = SimpleNamespace(
            pk=10,
            question_text="Which option is correct?",
            answer_set=SimpleNamespace(all=lambda: answers),
        )
        second_question = SimpleNamespace(
            pk=13,
            question_text="Which is another option?",
            answer_set=SimpleNamespace(all=lambda: answers),
        )
        third_question = SimpleNamespace(
            pk=14,
            question_text="Which is the final option?",
            answer_set=SimpleNamespace(all=lambda: answers),
        )
        quiz = SimpleNamespace(
            pk=8,
            question_set=SimpleNamespace(all=lambda: [question, second_question]),
        )
        older_quiz = SimpleNamespace(
            pk=7,
            question_set=SimpleNamespace(all=lambda: [third_question]),
        )
        manual_question = SimpleNamespace(
            pk=16,
            question_text="A manually created question",
            answer_set=SimpleNamespace(all=lambda: answers),
        )
        manual_quiz = SimpleNamespace(
            pk=6,
            question_set=SimpleNamespace(all=lambda: [manual_question]),
        )
        generated_quizzes = [quiz, older_quiz]
        quiz_queryset = SimpleNamespace(
            prefetch_related=lambda *_args: generated_quizzes
        )
        lesson = SimpleNamespace(
            pk=9,
            title="Django lesson",
            description="A lesson about Django.",
            course=SimpleNamespace(title="Django course"),
            quizzes=SimpleNamespace(
                filter=Mock(
                    side_effect=lambda **kwargs: quiz_queryset
                    if kwargs == {"is_ai_generated": True}
                    else SimpleNamespace(
                        prefetch_related=lambda *_args: SimpleNamespace(
                            all=lambda: [manual_quiz]
                        )
                    )
                )
            ),
            video_url="https://example.com/lesson",
        )
        get_lesson.return_value = lesson

        response = self.client.get(
            reverse("video_lesson_detail", kwargs={"lesson_id": lesson.pk})
        )

        self.assertContains(response, "Which option is correct?")
        self.assertContains(response, "Which is another option?")
        self.assertContains(response, "Which is the final option?")
        self.assertNotContains(response, "A manually created question")
        lesson.quizzes.filter.assert_called_once_with(is_ai_generated=True)
        self.assertContains(response, "Option one")
        self.assertContains(response, "Option two")
        rendered = response.content.decode()
        self.assertEqual(rendered.count('action="/quizzes/8/submit/"'), 1)
        self.assertEqual(rendered.count(">Submit all answers</button>"), 1)

    async def test_successful_generation_saves_and_redirects(self):
        lesson = SimpleNamespace(pk=9, description="A lesson about Django.")
        questions = [
            {
                "question": "What is Django?",
                "options": ["Framework", "Database", "Browser", "Editor"],
                "correctAnswer": "Framework",
            }
        ]
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("lms.views._get_video_lesson", return_value=lesson),
            patch("lms.views._request_gemini", new_callable=AsyncMock) as request,
            patch("lms.views._save_quiz_questions") as save,
        ):
            request.return_value = questions
            response = await self.async_client.post(
                reverse("generate_quiz", kwargs={"lesson_id": lesson.pk})
            )

        save.assert_called_once_with(lesson, questions)
        self.assertEqual(
            response["Location"],
            f"{reverse('video_lesson_detail', args=[lesson.pk])}?quiz_generated=1",
        )

    async def test_generation_requires_post(self):
        response = await self.async_client.get(
            reverse("generate_quiz", kwargs={"lesson_id": 9})
        )

        self.assertEqual(response.status_code, 405)
        self.assertIn("error", response.json())

    async def test_generation_requires_api_key(self):
        lesson = SimpleNamespace(pk=9, description="A lesson about Django.")
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
            patch("lms.views._get_video_lesson", return_value=lesson),
        ):
            response = await self.async_client.post(
                reverse("generate_quiz", kwargs={"lesson_id": lesson.pk})
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("quiz_error=", response["Location"])
        self.assertIn("GEMINI_API_KEY", response["Location"])

    async def test_generation_redirects_back_with_api_error(self):
        lesson = SimpleNamespace(pk=9, description="A lesson about Django.")
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("lms.views._get_video_lesson", return_value=lesson),
            patch(
                "lms.views._request_gemini",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Service unavailable."),
            ),
        ):
            response = await self.async_client.post(
                reverse("generate_quiz", kwargs={"lesson_id": lesson.pk})
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("quiz_error=Service+unavailable.", response["Location"])

    async def test_generation_redirects_back_with_database_error(self):
        lesson = SimpleNamespace(pk=9, description="A lesson about Django.")
        questions = [{"question": "Question?", "options": [], "correctAnswer": ""}]
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("lms.views._get_video_lesson", return_value=lesson),
            patch(
                "lms.views._request_gemini",
                new_callable=AsyncMock,
                return_value=questions,
            ),
            patch(
                "lms.views._save_quiz_questions",
                side_effect=RuntimeError("Database unavailable."),
            ),
        ):
            response = await self.async_client.post(
                reverse("generate_quiz", kwargs={"lesson_id": lesson.pk})
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("quiz_error=The+generated+quiz+could+not+be+saved", response["Location"])

    def test_api_response_parser_rejects_malformed_question(self):
        response_body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        [
                                            {
                                                "question": "Question?",
                                                "options": ["Only one"],
                                                "correctAnswer": "Only one",
                                            }
                                        ]
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode()

        with patch("lms.views._send_gemini_request", return_value=response_body):
            with self.assertRaises(ValueError):
                asyncio.run(_request_gemini("test-key", "Lesson description"))

    def test_quiz_save_creates_one_quiz_with_all_questions(self):
        course = Course(title="Course", description="Course description")
        lesson = VideoLesson(course=course)
        quiz = Quiz(course=course)
        question = Question(quiz=quiz)
        questions = [
            {
                "question": "Question one?",
                "options": ["A", "B", "C", "D"],
                "correctAnswer": "B",
            },
            {
                "question": "Question two?",
                "options": ["E", "F", "G", "H"],
                "correctAnswer": "G",
            },
        ]
        with (
            patch("lms.views.transaction.atomic"),
            patch("lms.views.Quiz.objects.create", return_value=quiz) as create_quiz,
            patch(
                "lms.views.Question.objects.create", return_value=question
            ) as create_question,
            patch("lms.views.Answer.objects.bulk_create") as create_answers,
        ):
            _save_quiz_questions(lesson, questions)

        create_quiz.assert_called_once_with(
            course=lesson.course,
            lesson=lesson,
            is_ai_generated=True,
        )
        self.assertEqual(create_question.call_count, 2)
        self.assertTrue(
            all(call.kwargs["quiz"] is quiz for call in create_question.call_args_list)
        )
        self.assertEqual(create_answers.call_count, 2)
        first_answers = create_answers.call_args_list[0].args[0]
        self.assertEqual(
            [answer.is_correct for answer in first_answers],
            [False, True, False, False],
        )


class QuizSubmissionTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = SimpleNamespace(is_authenticated=True, pk=12)
        self.correct_answer = SimpleNamespace(
            pk=21, answer_text="Correct", is_correct=True
        )
        self.incorrect_answer = SimpleNamespace(
            pk=22, answer_text="Incorrect", is_correct=False
        )
        self.question = SimpleNamespace(pk=15, question_text="A question?")
        self.question.answer_set = SimpleNamespace(
            all=lambda: [self.correct_answer, self.incorrect_answer]
        )
        self.quiz = SimpleNamespace(pk=8)
        self.quiz.is_ai_generated = True
        self.quiz.course = SimpleNamespace(pk=3)
        self.quiz.lesson_id = 4
        self.quiz.lesson = SimpleNamespace(
            quizzes=SimpleNamespace(
                filter=lambda **_kwargs: [self.quiz]
            )
        )
        self.quiz.question_set = SimpleNamespace(
            prefetch_related=lambda *_args: SimpleNamespace(
                all=lambda: [self.question]
            )
        )

    def test_submit_scores_correct_answer_and_redirects(self):
        request = self.factory.post(
            reverse("quiz_submit", kwargs={"quiz_id": self.quiz.pk}),
            {str(self.question.pk): str(self.correct_answer.pk)},
        )
        request.user = self.user
        attempt = SimpleNamespace(pk=41)

        with (
            patch("lms.views.get_object_or_404", return_value=self.quiz),
            patch("lms.views.QuizAttempt.objects.create", return_value=attempt) as save,
            patch("lms.views.CourseCompletion.objects.get_or_create") as get_completion,
        ):
            completion = SimpleNamespace(is_completed=False, save=Mock())
            get_completion.return_value = (completion, True)
            response = quiz_submit(request, quiz_id=self.quiz.pk)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response["Location"],
            reverse("quiz_results", kwargs={"attempt_id": attempt.pk}),
        )
        save.assert_called_once_with(
            user=self.user,
            quiz=self.quiz,
            score=1,
            results={
                str(self.question.pk): {
                    "selected_answer_id": self.correct_answer.pk,
                    "correct_answer_id": self.correct_answer.pk,
                    "is_correct": True,
                }
            },
        )
        get_completion.assert_called_once_with(
            user=self.user,
            course=self.quiz.course,
            defaults={"is_completed": False},
        )
        completion.save.assert_called_once_with(
            update_fields=["is_completed", "completed_at"]
        )
        self.assertIsNotNone(completion.completed_at)

    def test_submit_marks_missing_or_invalid_answer_incorrect(self):
        request = self.factory.post(
            reverse("quiz_submit", kwargs={"quiz_id": self.quiz.pk}),
            {str(self.question.pk): "999"},
        )
        request.user = self.user

        with (
            patch("lms.views.get_object_or_404", return_value=self.quiz),
            patch("lms.views.QuizAttempt.objects.create") as save,
            patch("lms.views.CourseCompletion.objects.get_or_create") as get_completion,
        ):
            save.return_value.pk = 42
            quiz_submit(request, quiz_id=self.quiz.pk)

        self.assertEqual(save.call_args.kwargs["score"], 0)
        get_completion.assert_not_called()
        self.assertEqual(
            save.call_args.kwargs["results"][str(self.question.pk)],
            {
                "selected_answer_id": None,
                "correct_answer_id": self.correct_answer.pk,
                "is_correct": False,
            },
        )

    def test_submit_grades_questions_from_every_quiz_for_lesson(self):
        other_correct = SimpleNamespace(pk=31, is_correct=True)
        other_incorrect = SimpleNamespace(pk=32, is_correct=False)
        other_question = SimpleNamespace(pk=25, question_text="Another question?")
        other_question.answer_set = SimpleNamespace(
            all=lambda: [other_correct, other_incorrect]
        )
        other_quiz = SimpleNamespace(
            question_set=SimpleNamespace(
                prefetch_related=lambda *_args: SimpleNamespace(
                    all=lambda: [other_question]
                )
            )
        )
        self.quiz.lesson.quizzes = SimpleNamespace(
            filter=lambda **_kwargs: [self.quiz, other_quiz]
        )
        request = self.factory.post(
            reverse("quiz_submit", kwargs={"quiz_id": self.quiz.pk}),
            {
                str(self.question.pk): str(self.correct_answer.pk),
                str(other_question.pk): str(other_incorrect.pk),
            },
        )
        request.user = self.user

        with (
            patch("lms.views.get_object_or_404", return_value=self.quiz),
            patch("lms.views.QuizAttempt.objects.create") as save,
            patch("lms.views.CourseCompletion.objects.get_or_create") as get_completion,
        ):
            save.return_value.pk = 42
            quiz_submit(request, quiz_id=self.quiz.pk)

        self.assertEqual(save.call_args.kwargs["score"], 1)
        self.assertEqual(len(save.call_args.kwargs["results"]), 2)
        get_completion.assert_not_called()

    def test_passing_threshold_is_80_percent(self):
        questions = []
        submitted_answers = {}
        for index in range(5):
            correct = SimpleNamespace(pk=100 + index * 2, is_correct=True)
            incorrect = SimpleNamespace(pk=101 + index * 2, is_correct=False)
            question = SimpleNamespace(pk=200 + index)
            question.answer_set = SimpleNamespace(all=lambda c=correct, i=incorrect: [c, i])
            questions.append(question)
            submitted_answers[str(question.pk)] = str(
                correct.pk if index < 4 else incorrect.pk
            )
        self.quiz.question_set = SimpleNamespace(
            prefetch_related=lambda *_args: SimpleNamespace(all=lambda: questions)
        )
        request = self.factory.post(
            reverse("quiz_submit", kwargs={"quiz_id": self.quiz.pk}),
            submitted_answers,
        )
        request.user = self.user

        with (
            patch("lms.views.get_object_or_404", return_value=self.quiz),
            patch("lms.views.QuizAttempt.objects.create") as save_attempt,
            patch("lms.views.CourseCompletion.objects.get_or_create") as get_completion,
        ):
            save_attempt.return_value.pk = 42
            get_completion.return_value = (
                SimpleNamespace(is_completed=True),
                False,
            )
            quiz_submit(request, quiz_id=self.quiz.pk)

        self.assertEqual(save_attempt.call_args.kwargs["score"], 4)
        get_completion.assert_called_once()

    def test_score_below_80_percent_does_not_complete_course(self):
        questions = []
        submitted_answers = {}
        for index in range(5):
            correct = SimpleNamespace(pk=300 + index * 2, is_correct=True)
            incorrect = SimpleNamespace(pk=301 + index * 2, is_correct=False)
            question = SimpleNamespace(pk=400 + index)
            question.answer_set = SimpleNamespace(all=lambda c=correct, i=incorrect: [c, i])
            questions.append(question)
            submitted_answers[str(question.pk)] = str(
                correct.pk if index < 3 else incorrect.pk
            )
        self.quiz.question_set = SimpleNamespace(
            prefetch_related=lambda *_args: SimpleNamespace(all=lambda: questions)
        )
        request = self.factory.post(
            reverse("quiz_submit", kwargs={"quiz_id": self.quiz.pk}),
            submitted_answers,
        )
        request.user = self.user

        with (
            patch("lms.views.get_object_or_404", return_value=self.quiz),
            patch("lms.views.QuizAttempt.objects.create") as save_attempt,
            patch("lms.views.CourseCompletion.objects.get_or_create") as get_completion,
        ):
            save_attempt.return_value.pk = 42
            quiz_submit(request, quiz_id=self.quiz.pk)

        self.assertEqual(save_attempt.call_args.kwargs["score"], 3)
        get_completion.assert_not_called()

    def test_submit_rejects_non_post_and_anonymous_requests(self):
        get_request = self.factory.get(
            reverse("quiz_submit", kwargs={"quiz_id": self.quiz.pk})
        )
        get_request.user = self.user
        self.assertEqual(quiz_submit(get_request, self.quiz.pk).status_code, 405)

        post_request = self.factory.post(
            reverse("quiz_submit", kwargs={"quiz_id": self.quiz.pk})
        )
        post_request.user = SimpleNamespace(is_authenticated=False)
        response = quiz_submit(post_request, self.quiz.pk)
        self.assertEqual(response.status_code, 401)

    def test_results_are_limited_to_attempt_owner(self):
        attempt = SimpleNamespace(
            pk=41,
            user=self.user,
            quiz=self.quiz,
            score=1,
            results={
                str(self.question.pk): {
                    "selected_answer_id": self.correct_answer.pk,
                    "correct_answer_id": self.correct_answer.pk,
                    "is_correct": True,
                }
            },
        )
        request = self.factory.get(
            reverse("quiz_results", kwargs={"attempt_id": attempt.pk})
        )
        request.user = self.user

        with (
            patch("lms.views.QuizAttempt.objects.select_related"),
            patch("lms.views.get_object_or_404", return_value=attempt) as get_attempt,
            patch("lms.views.CourseCompletion.objects.filter") as find_completion,
        ):
            find_completion.return_value.exists.return_value = True
            response = quiz_results(request, attempt_id=attempt.pk)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your score: 1")
        self.assertContains(response, "Course completed!")
        self.assertContains(response, "Download your certificate")
        self.assertContains(
            response,
            reverse(
                "generate_certificate",
                kwargs={
                    "course_id": self.quiz.course.pk,
                    "user_id": self.user.pk,
                },
            ),
        )
        self.assertContains(response, "Correct")
        self.assertEqual(get_attempt.call_args.kwargs["user"], self.user)
        find_completion.assert_called_once_with(
            user=self.user,
            course=self.quiz.course,
            is_completed=True,
        )

    @patch("lms.views.get_object_or_404")
    def test_course_detail_displays_signed_in_users_completion(self, get_course):
        course = SimpleNamespace(
            pk=7,
            title="Django Basics",
            description="An introduction to Django.",
            videolesson_set=SimpleNamespace(all=lambda: []),
        )
        get_course.return_value = course
        request = self.factory.get(reverse("course_detail", kwargs={"course_id": 7}))
        request.user = self.user

        with patch("lms.views.CourseCompletion.objects.filter") as find_completion:
            find_completion.return_value.exists.return_value = True
            response = course_detail(request, course_id=7)

        self.assertContains(response, "Course Completed!")
        self.assertContains(response, "Download Certificate")
        self.assertContains(
            response,
            reverse(
                "generate_certificate",
                kwargs={"course_id": course.pk, "user_id": self.user.pk},
            ),
        )
        find_completion.assert_called_once_with(
            user=self.user,
            course=course,
            is_completed=True,
        )

    def test_generate_certificate_downloads_completed_course_as_pdf(self):
        user = SimpleNamespace(
            pk=12,
            is_authenticated=True,
            get_full_name=lambda: "Alex Example",
            get_username=lambda: "alex",
        )
        course = SimpleNamespace(pk=3, title="Django Basics")
        completion = SimpleNamespace(
            user=user,
            course=course,
            course_id=course.pk,
            completed_at=datetime(2025, 1, 2, tzinfo=datetime_timezone.utc),
        )
        request = self.factory.get(
            reverse(
                "generate_certificate",
                kwargs={"course_id": course.pk, "user_id": user.pk},
            )
        )
        request.user = user

        with (
            patch("lms.views.CourseCompletion.objects.select_related") as select_related,
            patch("lms.views.get_object_or_404", return_value=completion) as get_completion,
        ):
            response = generate_certificate(
                request,
                course_id=course.pk,
                user_id=user.pk,
            )
            pdf_data = b"".join(response.streaming_content)

        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("attachment;", response["Content-Disposition"])
        self.assertTrue(pdf_data.startswith(b"%PDF-"))
        self.assertIn(b"Alex Example", pdf_data)
        self.assertIn(b"Django Basics", pdf_data)
        self.assertIn(b"January 02, 2025", pdf_data)
        select_related.assert_called_once_with("user", "course")
        get_completion.assert_called_once_with(
            select_related.return_value,
            user=user,
            course_id=course.pk,
            is_completed=True,
        )

    def test_generate_certificate_rejects_another_users_id(self):
        request = self.factory.get("/")
        request.user = SimpleNamespace(pk=12, is_authenticated=True)

        with self.assertRaises(Http404):
            generate_certificate(request, course_id=3, user_id=13)
