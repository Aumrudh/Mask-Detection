from django.urls import path,include
from . import views


def aoi():
  for i in range(10):
    if i%2:
      yield i

urlpatterns = [
    path('', views.hi,name='home-page'),
    path('add',views.add,name='add'),
    path('add2',views.add2,name='add2'),
    path('add3',views.add3,name='add3'),
        path('index', views.index, name='index'),
                path('x', views.x, name='x'),
    path('video_feed', views.video_feed, name='video_feed'),
     path('test_stream', views.test_stream, name='test_stream'),
     
 
]
