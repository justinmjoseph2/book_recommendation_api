from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BookViewSet, RecommendationViewSet
from sklearn.feature_extraction.text import TfidfVectorizer

router = DefaultRouter()
router.register(r'books', BookViewSet)
router.register(r'recommendations', RecommendationViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('books/<int:pk>/recommend/', BookViewSet.as_view({'get': 'recommend'}), name='book-recommend'),
]
