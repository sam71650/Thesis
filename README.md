# Master Thesis
# Face Recognition System for Master Thesis

This repo is created for Master Thesis Spring 2026 at Stockholm University 
This project implements a **Django-based face recognition system** with user authentication and real-time database integration.  
It demonstrates **building, testing, and containerizing a Django application** using **GitHub Actions CI** and **Docker**.  

---

## Overview

| Component | Description |
|-----------|-------------|
| **Django App** | Core face recognition system with `users` and authentication modules |
| **Database** | PostgreSQL service configured via GitHub Actions |
| **CI/CD** | Automated testing, migrations, and Docker image build via GitHub Actions |

---

## Repository Structure

face_recognition_system/
│
├── .github/workflows/ # CI/CD workflow for Django app
├── face_recognition_system/ # Django project code
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── users/ # Django app for user management
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   ├── views.py
│   └── migrations/
├── templates/ # HTML templates
│   ├── base.html
│   ├── login.html
│   └── register.html
├── manage.py
├── requirements.txt
├── Dockerfile # Docker build for deployment
├── .env # Local environment variables (not committed)
└── README.md

---

## Requirements

- Python 3.11+  
- Django 4.x  
- PostgreSQL  
- psycopg2-binary  
- Docker  
Plese Find All Related Packages in `requirements.txt`
---

## Environment Variables

Create a `.env` file in the project root with:

```bash
POSTGRES_USER=postgres
POSTGRES_PASSWORD=YOUR_PASSWORD
POSTGRES_DB=postgres
SECRET_KEY='your-django-secret-key'
DEBUG=True
```

## Docker Image

You can run the application using the Docker image.

### Pull Image

```bash
docker pull ghcr.io/sam71650/face_recognition_master_thesis:latest
```
