from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    embedding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)
