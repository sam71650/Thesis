from django.contrib.auth.models import AbstractUser
from django.db import models

class Person(AbstractUser):
    embedding  = models.BinaryField(null=True, blank=True)
    face_image = models.ImageField(upload_to="faces/", null=True, blank=True)

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='customuser_set',
        blank=True
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='customuser_set',
        blank=True
    )

    def __str__(self):
        return self.username