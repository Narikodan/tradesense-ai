"""
Django settings for TradeSense AI project.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INSECURE_SECRET_KEY = 'django-insecure-change-this-in-production!'
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', DEFAULT_INSECURE_SECRET_KEY)
DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() in {'1', 'true', 'yes', 'on'}
ALLOWED_HOSTS = [host.strip() for host in os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',') if host.strip()]

if not DEBUG and SECRET_KEY == DEFAULT_INSECURE_SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY must be set to a non-default value when DJANGO_DEBUG=False.")

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Local apps
    'core',
    'stocks',
    'analysis',  # not a Django app, but we can still register it for management commands
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'tradesense_ai.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'tradesense_ai.wsgi.application'

# Database – SQLite for development (switch to PostgreSQL for production)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Cache - database cache locally, optional Redis in production.
REDIS_URL = os.environ.get('DJANGO_REDIS_URL', '').strip()
if REDIS_URL:
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.redis.RedisCache',
            'LOCATION': REDIS_URL,
        }
    }
else:
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
            'LOCATION': os.environ.get('DJANGO_DB_CACHE_TABLE', 'tradesense_cache'),
        }
    }

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']  # for global static, but we'll use app-level
STATIC_ROOT = os.environ.get('DJANGO_STATIC_ROOT', str(BASE_DIR / 'staticfiles'))

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https') if os.environ.get('DJANGO_SECURE_PROXY_SSL_HEADER', '').lower() in {'1', 'true', 'yes', 'on'} else None
SECURE_SSL_REDIRECT = os.environ.get('DJANGO_SECURE_SSL_REDIRECT', 'False').lower() in {'1', 'true', 'yes', 'on'}
SESSION_COOKIE_SECURE = os.environ.get('DJANGO_SESSION_COOKIE_SECURE', str(not DEBUG)).lower() in {'1', 'true', 'yes', 'on'}
CSRF_COOKIE_SECURE = os.environ.get('DJANGO_CSRF_COOKIE_SECURE', str(not DEBUG)).lower() in {'1', 'true', 'yes', 'on'}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': os.environ.get('DJANGO_LOG_LEVEL', 'INFO'),
    },
}
