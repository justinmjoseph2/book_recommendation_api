from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import Book, Recommendation
from .serializers import BookSerializer, RecommendationSerializer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        # Retrieve 5 recommendations for the book
        recommendations = Recommendation.objects.filter(book=instance)[:5]
        recommendations_serializer = RecommendationSerializer(recommendations, many=True)
        data = {
            'book': serializer.data,
            'recommendations': recommendations_serializer.data
        }
        return Response(data)
    
    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    def recommend(self, request, pk=None):
        # Get the book
        book = self.get_object()

        # Get all books excluding the current one
        all_books = Book.objects.exclude(id=book.id)

        # Create TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')

        # Combine author and description for each book
        book_descriptions = [f"{book.author} {book.description}" for book in all_books]
        book_descriptions.append(f"{book.author} {book.description}")  # Add current book

        # Compute TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(book_descriptions)

        # Compute cosine similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Get indices of similar books
        similar_indices = np.argsort(cosine_sim[-1])[::-1][1:6]  # Exclude the current book and get top 5

        # Ensure similar_indices is a list of integers
        similar_indices = list(map(int, similar_indices))

        # Get recommended books based on similar authors or descriptions
        recommended_books = [all_books[idx] for idx in similar_indices if
                             all_books[idx].author.lower() == book.author.lower()
                             or cosine_sim[-1][idx] > 0.2]  # Adjust the threshold as needed

        # If no books found based on author or description similarity, fallback to top 5 similar books
        if not recommended_books:
            recommended_books = [all_books[idx] for idx in similar_indices]

        # Serialize recommended books with images and details
        data = []
        for recommended_book in recommended_books:
            book_data = {
                'title': recommended_book.title,
                'author': recommended_book.author,
                'description': recommended_book.description,
                'image_url': request.build_absolute_uri(recommended_book.image.url) if recommended_book.image else None,
            }
            data.append(book_data)

        return Response(data)
    
    
class RecommendationViewSet(viewsets.ModelViewSet):
    queryset = Recommendation.objects.all()
    serializer_class = RecommendationSerializer

    # This is the list action to retrieve recommendations for a specific book
    def list(self, request, pk=None):
        queryset = self.queryset.filter(book_id=pk)
        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)